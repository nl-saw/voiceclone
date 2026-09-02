"""Chatterbox Multilingual V3 engine (Resemble AI) — high-quality zero-shot cloning.

* MIT-licensed code AND weights; very actively maintained upstream
  (repo: https://github.com/resemble-ai/chatterbox, pip: ``chatterbox-tts``).
* 23 languages for zero-shot cloning **including Dutch** — the broadest coverage
  in this toolkit. Emotion is expressed through Chatterbox's "exaggeration"
  intensity dial (see workers/chatterbox_worker.py for the mapping).
* No official fine-tuning recipe yet, so ``finetune=False`` for now.
* Runs in a dedicated venv (upstream pins torch==2.6.0, transformers==5.2.0;
  we upgrade torch to 2.9.1 post-install for Blackwell/RTX-50xx support) that
  conflicts with the toolkit's own dependencies → external-engine worker protocol.
* Outputs carry Resemble's imperceptible PerTh neural watermark.

Install:   voiceclone install-engine chatterbox
Synthesize: voiceclone synthesize <voice> --engine chatterbox "text"
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from .external import ExternalEngine, ExternalEngineError

# V3 multilingual (23 languages incl. Dutch) is not on PyPI yet — the newest
# release (0.1.7) only ships the legacy V2 checkpoint. Install from git, pinned
# to a known-good commit. To upgrade: bump GIT_REF and delete
# data/engines/chatterbox/installed.json (then re-run install-engine).
GIT_REF = "5de7a54aa4e5"  # master @ 2026-07-21 (V3 + single-language pack)
PIP_PACKAGE = f"chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@{GIT_REF}"
PYTHON_VERSION = "3.10"  # upstream supports 3.10+

# Upstream pins torch==2.6.0 for Python < 3.14, but that build predates NVIDIA
# Blackwell (sm_120): on RTX 50xx, inference dies with
# "CUDA error: no kernel image is available for execution on the device".
# Upstream itself declares torch>=2.9.0 fine (that is what its Python >= 3.14
# branch installs), so we upgrade after installing the package; torchaudio must
# track torch's version. Bump these together when upgrading further.
TORCH_PIN = "torch==2.9.1"
TORCHAUDIO_PIN = "torchaudio==2.9.1"


class ChatterboxEngine(ExternalEngine):
    name = "chatterbox"

    def __init__(self) -> None:
        super().__init__()
        self.worker_script = Path(__file__).resolve().parent / "workers" / "chatterbox_worker.py"

    def extra_env(self) -> dict[str, str]:
        # keep the HF model cache inside the per-engine state dir
        return {"HF_HOME": str(self.engine_dir / "hf")}


def _run_streaming(cmd: list[str], logline, cwd: str | None = None) -> None:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd
    )
    assert proc.stdout is not None
    tail: list[str] = []
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
        if any(k in line for k in ("Collecting", "Installing", "Downloading", "error", "Error")):
            logline(line[:200])
    code = proc.wait()
    if code != 0:
        raise RuntimeError(
            f"Command failed (exit {code}): {' '.join(str(c) for c in cmd[:4])} ...\n"
            + "\n".join(tail[-25:])
        )


def _ensure_torch(eng: ChatterboxEngine, logline) -> None:
    """Upgrade torch/torchaudio past the upstream 2.6.0 pin (Blackwell fix).

    Idempotent — a fast no-op when the pins are already satisfied; also repairs
    venvs installed before this step existed.
    """
    venv_py = eng.venv_python()
    if not venv_py.exists():
        return
    uv = shutil.which("uv")
    cmd = (
        [uv, "pip", "install", "--python", str(venv_py)] if uv
        else [str(venv_py), "-m", "pip", "install"]
    )
    logline(f"Ensuring {TORCH_PIN.split('==')[0]} >= 2.9 for Blackwell GPUs (upstream pins 2.6.0) ...")
    _run_streaming(cmd + [TORCH_PIN, TORCHAUDIO_PIN], logline)


def ensure_installed(logline=print) -> None:
    """Create the dedicated venv and install ``chatterbox-tts``.

    Idempotent. Weights (~3 GB) download automatically from Hugging Face on the
    first synthesis (cached under data/engines/chatterbox/hf).
    """
    eng = ChatterboxEngine()
    engine_dir = eng.engine_dir
    engine_dir.mkdir(parents=True, exist_ok=True)
    if (engine_dir / "installed.json").exists():
        _ensure_torch(eng, logline)  # repairs venvs from before the fix
        logline("Already installed.")
        return

    venv_py = eng.venv_python()
    uv = shutil.which("uv")
    if not venv_py.exists():
        if uv:
            logline(f"Creating Python {PYTHON_VERSION} venv with uv (interpreter may be downloaded) ...")
            _run_streaming([uv, "venv", "--python", PYTHON_VERSION, str(engine_dir / "venv")], logline)
        else:
            py = shutil.which(f"python{PYTHON_VERSION}") or shutil.which("python3.10")
            if not py:
                raise RuntimeError(
                    f"Need Python {PYTHON_VERSION} for Chatterbox (or 'uv' to fetch it). "
                    "Install uv: https://docs.astral.sh/uv/"
                )
            logline(f"Creating venv with {py} ...")
            _run_streaming([py, "-m", "venv", str(engine_dir / "venv")], logline)

    # dependencies (idempotent; fast no-op when already satisfied). Upstream's
    # torch==2.6.0 pin comes in here and is replaced by _ensure_torch below.
    logline(f"Installing {PIP_PACKAGE} — this takes a while")
    if uv:
        _run_streaming([uv, "pip", "install", "--python", str(venv_py), PIP_PACKAGE], logline)
    else:
        _run_streaming([str(venv_py), "-m", "pip", "install", "-U", "pip"], logline)
        _run_streaming([str(venv_py), "-m", "pip", "install", PIP_PACKAGE], logline)

    _ensure_torch(eng, logline)

    (engine_dir / "installed.json").write_text(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package": PIP_PACKAGE,
        "python": PYTHON_VERSION,
    }))
