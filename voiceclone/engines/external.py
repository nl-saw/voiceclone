"""Subprocess-isolated engines.

Some TTS engines cannot share the toolkit's Python environment (hard-pinned
torch/transformers versions, a different required Python minor version). Those
run in a dedicated venv under ``data/engines/<name>/venv`` and are driven
through a small JSON-lines *worker* process:

  * one JSON command per line on the worker's stdin,
  * one JSON response per line on its stdout,
  * the model is loaded once inside the worker and kept warm between calls.

Protocol (JSON lines):
  request   {"op": "ping"}
  request   {"op": "synthesize", "text", "reference_wav_path", "reference_text",
             "language", "emotion", "style", "finetuned_checkpoint"}
  response  {"ok": true, "wav_path", "sample_rate", "mode"} | {"ok": false, "error"}

The worker's first stdout line is a readiness handshake:
  {"ok": true, "op": "ready", "engine": "<name>"}

Each concrete external engine module provides:

  * a worker script (``voiceclone/engines/workers/<name>_worker.py``) that
    implements the protocol using the engine's own API — it runs with the
    ENGINE's venv python, not ours;
  * ``ensure_installed()`` — clone repo + create venv + install dependencies
    (heavy; invoked by ``voiceclone install-engine <name>``);
  * optionally ``finetune(...)`` for fine-tuning.

Configuration reaches the worker through environment variables (VC_ENGINE_DIR,
VC_MODEL_DIR, ...) set at spawn time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..config import get_settings
from .base import Engine, SynthesisResult


class ExternalEngineError(Exception):
    pass


def installer_env() -> dict[str, str]:
    """Environment for install subprocesses that keeps every cache in the project.

    By default uv and pip write to the user's home directory (``~/.cache/uv`` alone
    easily reaches 10+ GB during an engine install) and temp files go to ``/tmp`` —
    on a machine with a small home partition or a different drive that is not the
    project's, installs die mid-way with "no space left". Redirect everything under
    ``data/cache`` (shared across engines, so e.g. torch wheels are cached once) and
    temp files to ``data/tmp``. Also defaults ``UV_LINK_MODE=copy`` on filesystems
    without hardlink support. Values already set by the user take precedence.
    """
    s = get_settings()
    env = os.environ.copy()
    for key, sub in (
        ("UV_CACHE_DIR", "uv"),
        ("UV_PYTHON_INSTALL_DIR", "uv-python"),
        ("PIP_CACHE_DIR", "pip"),
        ("HF_HOME", "hf"),
    ):
        if key not in os.environ:
            env[key] = str(s.cache_dir / sub)
    if not os.environ.get("TMPDIR"):
        tmp = s.data_dir / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(tmp)
    # On filesystems without hardlink support (NFS/SMB/CIFS shares, some overlays)
    # uv warns "Failed to hardlink files; falling back to full copy" for every
    # install. Probe the project drive once; if hardlinks are impossible there,
    # tell uv up front so it copies quietly instead of warning each time.
    if not os.environ.get("UV_LINK_MODE") and not _hardlinks_supported(s.data_dir):
        env["UV_LINK_MODE"] = "copy"
    return env


def _hardlinks_supported(d: Path) -> bool:
    """Whether the filesystem holding *d* supports hardlinks (uv's fast install path)."""
    a, b = d / ".hl-probe-a", d / ".hl-probe-b"
    try:
        d.mkdir(parents=True, exist_ok=True)
        a.write_bytes(b"x")
        os.link(str(a), str(b))
        return True
    except OSError:
        return False
    finally:
        for p in (a, b):
            p.unlink(missing_ok=True)


class ExternalEngine(Engine):
    """Base class for engines running in their own venv via a worker process."""

    name: str = "external"
    #: absolute path of the worker script (set by subclasses)
    worker_script: Path | None = None

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()  # one in-flight command per worker

    # ------------------------------------------------------------------ #
    # environment
    # ------------------------------------------------------------------ #
    @property
    def engine_dir(self) -> Path:
        """Per-engine state dir: venv, cloned repo, models, logs."""
        return get_settings().data_dir / "engines" / self.name

    def venv_python(self) -> Path:
        return self.engine_dir / "venv" / "bin" / "python"

    def is_installed(self) -> bool:
        return self.venv_python().exists()

    def extra_env(self) -> dict[str, str]:
        """Extra environment variables for the worker (engine-specific)."""
        return {}

    # ------------------------------------------------------------------ #
    # worker lifecycle
    # ------------------------------------------------------------------ #
    def _log_path(self) -> Path:
        p = self.engine_dir / "worker.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _spawn(self) -> None:
        if not self.is_installed():
            raise ExternalEngineError(
                f"Engine '{self.name}' is not installed on this machine. "
                f"Run:  voiceclone install-engine {self.name}"
            )
        if self.worker_script is None or not Path(self.worker_script).exists():
            raise ExternalEngineError(f"Engine '{self.name}' has no worker script.")

        env = os.environ.copy()
        env["VC_ENGINE_DIR"] = str(self.engine_dir)
        env.update(self.extra_env())

        logf = open(self._log_path(), "a", encoding="utf-8")
        logf.write(f"\n===== worker start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        logf.flush()
        self._proc = subprocess.Popen(
            [str(self.venv_python()), str(self.worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=logf,
            env=env,
            text=True,
            bufsize=1,
        )
        ready = self._read_response()
        if not ready.get("ok"):
            raise ExternalEngineError(
                f"Engine '{self.name}' worker failed to start: {ready.get('error', 'unknown')}"
            )

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._spawn()

    def _read_response(self) -> dict:
        """Read one protocol response, skipping non-protocol lines.

        Third-party C/Python libraries inside the worker (modelscope download
        progress, onnxruntime, triton ...) occasionally write to stdout; those
        lines are logged and skipped until a real JSON response arrives.
        """
        assert self._proc is not None
        for _ in range(100_000):
            line = self._proc.stdout.readline()
            if not line:
                code = self._proc.poll()
                raise ExternalEngineError(
                    f"Engine '{self.name}' worker exited (code={code}, see {self._log_path()})"
                )
            line = line.strip()
            if not line:
                continue
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                print(f"[{self.name}] worker stdout noise (skipped): {line[:150]}",
                      file=sys.stderr, flush=True)
                continue
            if isinstance(resp, dict) and "ok" in resp:
                return resp
        raise ExternalEngineError(
            f"Engine '{self.name}' worker produced no protocol response"
        )

    def _call(self, cmd: dict) -> dict:
        """Send one command, wait for its response. Thread-safe per engine."""
        with self._lock:
            self._ensure_worker()
            assert self._proc is not None
            try:
                self._proc.stdin.write(json.dumps(cmd) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError) as e:
                raise ExternalEngineError(
                    f"Engine '{self.name}' worker died mid-request (see {self._log_path()})"
                ) from e
            resp = self._read_response()
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "unknown engine error"))
        return resp

    # ------------------------------------------------------------------ #
    # Engine interface
    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        text: str,
        reference_wav_path: str,
        reference_text: str,
        language: str,
        emotion: str = "neutral",
        style: str | None = None,
        finetuned_checkpoint: str | None = None,
        temperature: float | None = None,
        length_penalty: float | None = None,
        repetition_penalty: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        speed: float | None = None,
    ) -> SynthesisResult:
        from .. import audio as A

        resp = self._call({
            "op": "synthesize",
            "text": text,
            "reference_wav_path": str(reference_wav_path),
            "reference_text": reference_text or "",
            "language": language,
            "emotion": emotion,
            "style": style,
            "finetuned_checkpoint": finetuned_checkpoint,
        })

        wav_path = Path(resp["wav_path"])
        sr = int(resp.get("sample_rate", 24000))
        wav = A.load_audio(str(wav_path), sr)  # decode at native rate (passthrough)
        wav_path.unlink(missing_ok=True)

        return SynthesisResult(
            wav=wav,
            sample_rate=sr,
            reference_file=str(reference_wav_path),
            reference_emotion=emotion,
            matched_requested_emotion=True,
            engine=self.name,
            mode=resp.get("mode", "zero-shot"),
            device=None,
        )

    def warmup(self) -> None:
        """Spawn the worker (model loads lazily on first synthesis)."""
        with self._lock:
            self._ensure_worker()
