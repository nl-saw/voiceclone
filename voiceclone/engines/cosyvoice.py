"""CosyVoice 3 engine (FunAudioLLM) — zero-shot cloning + official fine-tuning.

* Apache-2.0 code AND weights; actively maintained upstream
  (repo: https://github.com/FunAudioLLM/CosyVoice, model: Fun-CosyVoice3-0.5B-2512).
* 9 languages for zero-shot cloning: zh/en/ja/ko/de/es/fr/it/ru — **no Dutch**.
* Runs in a dedicated venv (upstream hard-pins torch==2.3.1, transformers==4.51.3,
  Python 3.10; we install a Blackwell-capable torch 2.9.1+cu128 instead) that
  conflicts with the toolkit's own dependencies, so it is driven through the
  external-engine worker protocol (see voiceclone/engines/external.py).

Install:   voiceclone install-engine cosyvoice3
Fine-tune: voiceclone train <voice> --engine cosyvoice3
           (official recipe: Kaldi-style data → parquet → torchrun train.py on the
           LLM component → averaged checkpoint → symlink-farm model dir)
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..config import get_settings
from .base import SynthesisResult  # noqa: F401 — re-exported for typing
from .external import ExternalEngine, ExternalEngineError, installer_env

REPO_URL = "https://github.com/FunAudioLLM/CosyVoice.git"
HF_MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
MODEL_DIRNAME = "Fun-CosyVoice3-0.5B"
PYTHON_VERSION = "3.10"  # upstream-recommended
INSTRUCT_PREFIX = "You are a helpful assistant.<|endofprompt|"


# --------------------------------------------------------------------------- #
# engine class (worker protocol)
# --------------------------------------------------------------------------- #

class CosyVoice3Engine(ExternalEngine):
    name = "cosyvoice3"

    def __init__(self) -> None:
        super().__init__()
        self.worker_script = Path(__file__).resolve().parent / "workers" / "cosyvoice3_worker.py"

    @property
    def model_dir(self) -> Path:
        return self.engine_dir / "models" / MODEL_DIRNAME

    def _ensure_worker(self):
        # Weights (~6 GB) auto-download on first use, as the README promises.
        # Without this hook a fresh machine fails inside the worker with
        # modelscope's "The request model: <local path> does not exist!" —
        # CosyVoice's AutoModel treats a missing local dir as a hub model id.
        # No-op (one filesystem check) once llm.pt is present.
        ensure_weights(lambda m: print(f"[cosyvoice3] {m}", file=sys.stderr, flush=True))
        super()._ensure_worker()


def _engine() -> CosyVoice3Engine:
    return CosyVoice3Engine()


# --------------------------------------------------------------------------- #
# installation (repo + venv + deps + weights)
# --------------------------------------------------------------------------- #

def _run_streaming(cmd: list[str], logline, cwd: str | None = None, env: dict | None = None) -> None:
    """Run a long command, streaming its output through logline; raise on failure."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd, env=env
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
        # keep the console uncluttered: only show progress-ish lines
        if any(k in line for k in ("Collecting", "Installing", "Downloading", "error", "Error", "ERROR")):
            logline(line[:200])
    code = proc.wait()
    if code != 0:
        raise RuntimeError(
            f"Command failed (exit {code}): {' '.join(str(c) for c in cmd[:4])} ...\n"
            + "\n".join(tail[-25:])
        )


# Upstream pins torch==2.3.1 (cu121), which predates NVIDIA Blackwell (sm_120):
# on RTX 50xx, inference dies with "CUDA error: no kernel image is available
# for execution on the device". We install a cu128 build instead (see
# _filtered_requirements). torchaudio must track torch. NOTE: torchaudio >= 2.9
# routes its native load/save through torchcodec, which needs a system FFmpeg
# of a matching major version — too fragile for "any machine", so the repo's
# two I/O call sites are patched to soundfile instead (_patch_repo_io).
# Verified end-to-end on CPU before shipping. Bump these together when
# upgrading further.
TORCH_PIN = "torch==2.9.1"
TORCHAUDIO_PIN = "torchaudio==2.9.1"


def _ensure_torch(eng: CosyVoice3Engine, logline) -> None:
    """Upgrade torch/torchaudio past the upstream 2.3.1 pin (Blackwell fix).

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
    logline(f"Ensuring torch >= 2.9 for Blackwell GPUs (upstream pins 2.3.1) ...")
    _run_streaming(cmd + [TORCH_PIN, TORCHAUDIO_PIN], logline, env=installer_env())


def _patch_deepspeed_compat(eng: CosyVoice3Engine, logline) -> None:
    """Make ``import deepspeed`` survive GPU boxes without a CUDA toolkit.

    The sdist build generates ``deepspeed/git_version_info.py``, which calls
    ``builder.is_compatible()`` for every op at import time. On a machine where
    torch sees a GPU but nvcc is missing, that raises MissingCUDAException —
    and transformers (which imports deepspeed whenever it is installed) becomes
    unimportable, killing CosyVoice inference. Wrap the probe so unavailable
    ops are simply reported as incompatible; CosyVoice trains with torch_ddp
    and never needs compiled deepspeed ops. Idempotent; also repairs venvs
    installed before this fix existed.
    """
    candidates = sorted((eng.engine_dir / "venv" / "lib").glob("python*/site-packages/deepspeed/git_version_info.py"))
    if not candidates:
        logline("deepspeed git_version_info.py not found — skipping compat patch")
        return
    f = candidates[0]
    src = f.read_text()
    old = (
        "for op_name, builder in ALL_OPS.items():\n"
        "    op_compatible = builder.is_compatible()\n"
        "    compatible_ops[op_name] = op_compatible\n"
    )
    new = (
        "for op_name, builder in ALL_OPS.items():\n"
        "    try:\n"
        "        op_compatible = builder.is_compatible()\n"
        "    except Exception:\n"
        "        # no CUDA toolkit / unsupported device: report the op as\n"
        "        # unavailable instead of failing 'import deepspeed'\n"
        "        op_compatible = False\n"
        "    compatible_ops[op_name] = op_compatible\n"
    )
    if old in src:
        f.write_text(src.replace(old, new))
        logline("Patched deepspeed import-time CUDA probe (git_version_info.py)")


def _patch_repo_io(repo: Path, logline) -> None:
    """Replace torchaudio file I/O in the cloned repo with soundfile.

    torchaudio >= 2.9 routes load/save through torchcodec, which needs a
    system FFmpeg of a matching major version — too fragile for "any machine"
    (and we only ever handle WAV). soundfile is already a hard dependency of
    the venv (via librosa) and needs no system libraries. Two call sites:
    reference-audio loading (inference) and dataset loading (fine-tuning).
    Idempotent; warns if upstream changed the lines.
    """
    patches = [
        ("cosyvoice/utils/file_utils.py",
         "def load_wav(wav, target_sr, min_sr=16000):\n"
         "    speech, sample_rate = torchaudio.load(wav, backend='soundfile')\n",
         "def load_wav(wav, target_sr, min_sr=16000):\n"
         "    import soundfile as _sf\n"
         "    import torch as _torch\n"
         "    _data, sample_rate = _sf.read(wav, dtype='float32', always_2d=True)\n"
         "    speech = _torch.from_numpy(_data).t().contiguous()  # channels-first, like torchaudio.load\n"),
        ("cosyvoice/dataset/processor.py",
         "        sample['speech'], sample['sample_rate'] = torchaudio.load(BytesIO(sample['audio_data']))\n",
         "        import soundfile as _sf\n"
         "        _data, _sr = _sf.read(BytesIO(sample['audio_data']), dtype='float32', always_2d=True)\n"
         "        sample['speech'] = torch.from_numpy(_data).t().contiguous()  # channels-first, like torchaudio.load\n"
         "        sample['sample_rate'] = _sr\n"),
    ]
    for rel, old, new in patches:
        f = repo / rel
        if not f.exists():
            logline(f"warning: {rel} not found — skipping I/O patch")
            continue
        src = f.read_text()
        if old in src:
            f.write_text(src.replace(old, new))
            logline(f"Patched torchaudio I/O -> soundfile ({rel})")
        elif "soundfile" not in src:
            logline(f"warning: expected pattern missing in {rel} — upstream may have changed; "
                    "torchaudio I/O would need torchcodec (system FFmpeg)")


def _patch_train_utils_join(repo: Path, logline) -> None:
    """Fix ``cosyvoice_join`` for torch >= 2.x.

    The repo reads the join-barrier timeout as ``group_join.options._timeout``,
    an attribute torch removed from ProcessGroup in 2.x. The resulting
    AttributeError is not a RuntimeError, so it escapes the "uneven workload"
    handler and kills every training run on the second batch. Fall back to
    monitored_barrier's default timeout (behavior-identical for single-GPU
    fine-tuning, where the barrier trivially succeeds). Idempotent; warns if
    upstream changed the lines.
    """
    f = repo / "cosyvoice" / "utils" / "train_utils.py"
    if not f.exists():
        logline("warning: cosyvoice/utils/train_utils.py not found — skipping join patch")
        return
    src = f.read_text()
    old = (
        "            dist.monitored_barrier(group=group_join,\n"
        "                                   timeout=group_join.options._timeout)\n"
    )
    new = (
        "            try:\n"
        "                _join_timeout = group_join.options._timeout  # torch < 2.x\n"
        "            except AttributeError:\n"
        "                _join_timeout = None  # torch >= 2.x removed ProcessGroup.options; default timeout\n"
        "            dist.monitored_barrier(group=group_join, timeout=_join_timeout)\n"
    )
    if old in src:
        f.write_text(src.replace(old, new))
        logline("Patched cosyvoice_join for torch >= 2.x (train_utils.py)")
    elif "_join_timeout" not in src:
        logline("warning: expected pattern missing in train_utils.py — upstream may have changed; "
                "training would crash with \"ProcessGroup ... has no attribute 'options'\"")


def _filtered_requirements(reqs: Path, engine_dir: Path) -> Path:
    """requirements.txt minus the torch/torchaudio pins.

    We install a newer Blackwell-capable torch build ourselves (see
    TORCH_PIN); keeping upstream's torch==2.3.1 pin in the same resolution
    would force a downgrade (and double-download ~6 GB of CUDA libraries).
    """
    out = engine_dir / "requirements.filtered.txt"
    kept = [l for l in reqs.read_text().splitlines() if not re.match(r"^\s*(torch|torchaudio)==", l)]
    out.write_text("\n".join(kept) + "\n")
    return out


def ensure_installed(logline=print) -> None:
    """Clone the repo, create the dedicated venv, install dependencies.

    Idempotent; safe to re-run (reuses existing repo/venv). Weights are NOT
    downloaded here — call :func:`ensure_weights` (or ``init --download``).
    """
    eng = _engine()
    engine_dir = eng.engine_dir
    engine_dir.mkdir(parents=True, exist_ok=True)
    venv_py = eng.venv_python()
    # The marker alone is not enough: the venv's bin/python is a symlink to the
    # uv-managed CPython, and if that interpreter was deleted (e.g. cleaning up
    # ~/.local/share/uv after a disk-space problem) the link dangles and the
    # engine is dead despite the marker. Fall through to the fresh path then —
    # it rebuilds the venv into the project's own cache dir (repo is reused).
    if (engine_dir / "installed.json").exists() and venv_py.exists():
        _ensure_torch(eng, logline)  # repairs venvs from before the fix
        _patch_repo_io(engine_dir / "repo", logline)  # same
        _patch_train_utils_join(engine_dir / "repo", logline)  # same
        _patch_deepspeed_compat(eng, logline)  # same
        logline("Already installed.")
        return

    # Fresh/repair path: drop any stale marker first. Without this, a run that
    # dies mid-way (e.g. disk full) after rebuilding the venv would leave a
    # marker claiming success while dependencies are still missing — the next
    # run would then short-circuit into "Already installed" with a broken engine.
    (engine_dir / "installed.json").unlink(missing_ok=True)

    # 1. repo -----------------------------------------------------------------
    repo = engine_dir / "repo"
    if not (repo / ".git").exists():
        logline(f"Cloning {REPO_URL} → {repo}")
        _run_streaming(
            ["git", "clone", "--recursive", REPO_URL, str(repo)], logline, env=installer_env()
        )
    elif not (repo / "third_party" / "Matcha-TTS").exists():
        logline("Updating git submodules (Matcha-TTS) ...")
        _run_streaming(["git", "-C", str(repo), "submodule", "update", "--init", "--recursive"], logline)
    _patch_repo_io(repo, logline)
    _patch_train_utils_join(repo, logline)

    # 2. venv (venv_py from the top of this function) -------------------------
    uv = shutil.which("uv")
    reqs = repo / "requirements.txt"
    if not reqs.exists():
        raise RuntimeError(f"{reqs} not found — is the repo clone complete?")
    env = installer_env()  # caches/temp inside the project, not the home dir
    if not venv_py.exists():
        if uv:
            logline(f"Creating Python {PYTHON_VERSION} venv with uv (interpreter may be downloaded) ...")
            # UV_VENV_CLEAR: replace the directory if a broken/old venv is still
            # there (newer uv refuses to overwrite without it; env form works on
            # all uv versions, unlike the --clear flag).
            _run_streaming(
                [uv, "venv", "--python", PYTHON_VERSION, str(engine_dir / "venv")], logline,
                env={**env, "UV_VENV_CLEAR": "1"},
            )
        else:
            py = shutil.which(f"python{PYTHON_VERSION}") or shutil.which("python3.10")
            if not py:
                raise RuntimeError(
                    f"Need Python {PYTHON_VERSION} for CosyVoice (or 'uv' to fetch it). "
                    "Install uv: https://docs.astral.sh/uv/"
                )
            logline(f"Creating venv with {py} ...")
            _run_streaming([py, "-m", "venv", str(engine_dir / "venv")], logline, env=env)

    # 3. dependencies (idempotent; fast no-op when already satisfied) ----------
    logline(f"Installing CosyVoice dependencies ({TORCH_PIN.split('==')[0]}, transformers 4.51.3, ...) — this takes a while")
    # deepspeed ships sdist-only on PyPI. Its setup.py probes the CUDA toolkit
    # (nvcc) whenever torch sees a GPU at *build* time and hard-fails with
    # MissingCUDAException otherwise — so on a GPU box without the CUDA
    # toolkit the install breaks. DS_ACCELERATOR=cpu makes setup.py build
    # deepspeed as pure Python (no compiled ops); that is all we need, since
    # CosyVoice fine-tuning runs with --train_engine torch_ddp and only uses
    # deepspeed for its Python-side utilities. Build-time env only — the
    # worker process never sees it. installer_env() keeps uv/pip caches and
    # temp files inside the project (see external.py).
    install_env = {**env, "DS_ACCELERATOR": "cpu"}
    if uv:
        # upstream requirements.txt pulls from official vendor indexes
        # (pytorch cu121, onnxruntime-cuda-12); uv needs the best-match
        # strategy to resolve across them. A few sdists (openai-whisper,
        # pyworld, deepspeed) import build-time deps that are not declared →
        # build with the venv's own environment (--no-build-isolation), which
        # means pre-installing setuptools<81 (newer dropped pkg_resources),
        # numpy and torch first.
        #
        # wheel-stub is needed too: tensorrt-cu12 / -libs on PyPI are tiny
        # "stub" sdists whose build backend (wheel_stub.buildapi) downloads
        # the real wheel from pypi.nvidia.com at build time (~4 GB for
        # tensorrt-cu12-libs). With --no-build-isolation that backend must be
        # importable from the venv itself, so it is pre-installed here.
        # torch/torchaudio come from our own pins (Blackwell-capable cu128
        # build), so resolve the rest of requirements.txt without them.
        filtered = _filtered_requirements(reqs, engine_dir)
        _run_streaming([uv, "pip", "install", "--python", str(venv_py), "setuptools<81", "wheel", "wheel-stub"], logline)
        _run_streaming([
            uv, "pip", "install", "--python", str(venv_py),
            TORCH_PIN, TORCHAUDIO_PIN, "numpy==1.26.4",
            "--index-strategy", "unsafe-best-match",
        ], logline, env=install_env)
        _run_streaming([
            uv, "pip", "install", "--python", str(venv_py), "-r", str(filtered),
            "--index-strategy", "unsafe-best-match", "--no-build-isolation",
        ], logline, env=install_env)
    else:
        _run_streaming([str(venv_py), "-m", "pip", "install", "-U", "pip"], logline)
        filtered = _filtered_requirements(reqs, engine_dir)
        _run_streaming([str(venv_py), "-m", "pip", "install", "-r", str(filtered),
                        TORCH_PIN, TORCHAUDIO_PIN], logline, env=install_env)

    _ensure_torch(eng, logline)  # no-op on fresh installs; keeps the pins authoritative
    _patch_deepspeed_compat(eng, logline)

    (engine_dir / "installed.json").write_text(json.dumps({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": REPO_URL,
        "python": PYTHON_VERSION,
    }))


def ensure_weights(logline=print) -> Path:
    """Download the Fun-CosyVoice3-0.5B-2512 weights (~6 GB) via huggingface_hub."""
    eng = _engine()
    model_dir = eng.model_dir
    if (model_dir / "llm.pt").exists():
        return model_dir
    if not eng.is_installed():
        raise ExternalEngineError(
            f"Engine '{eng.name}' is not installed. Run: voiceclone install-engine {eng.name}"
        )
    logline(f"Downloading {HF_MODEL_ID} (~6 GB) → {model_dir}")
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download({HF_MODEL_ID!r}, local_dir={str(model_dir)!r})"
    )
    _run_streaming([str(eng.venv_python()), "-c", code], logline, env=installer_env())
    if not (model_dir / "llm.pt").exists():
        raise RuntimeError(f"Download finished but {model_dir}/llm.pt is missing")
    return model_dir


# --------------------------------------------------------------------------- #
# fine-tuning (official CosyVoice 3 recipe, automated)
# --------------------------------------------------------------------------- #

def _read_pipe_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f, delimiter="|"):
            if len(r) >= 2 and r[0] not in ("audio_file",):
                rows.append({"audio_file": r[0], "text": r[1]})
    return rows


def _build_split(work: Path, split: str, voice_name: str, ds_dir: Path,
                 rows: list[dict]) -> int:
    """Lay out one split the way CosyVoice's prepare_data.py expects.

    <work>/data/<split>/<voice>/0001/<utt>.wav  (+ <utt>.normalized.txt)
    <work>/data/<split>/{wav.scp,text,utt2spk,spk2utt,instruct}
    """
    src = work / "data" / split
    wav_dir = src / voice_name / "0001"
    wav_dir.mkdir(parents=True, exist_ok=True)

    utt2wav, utt2text, utt2spk = {}, {}, {}
    n = 0
    for i, row in enumerate(rows):
        wav_src = ds_dir / row["audio_file"]
        if not wav_src.exists():
            continue
        utt = f"{voice_name}_{i:04d}"
        dst = wav_dir / f"{utt}.wav"
        shutil.copy2(wav_src, dst)
        (wav_dir / f"{utt}.normalized.txt").write_text(row["text"].strip() + "\n", encoding="utf-8")
        utt2wav[utt] = str(dst)
        utt2text[utt] = row["text"].strip()
        utt2spk[utt] = voice_name
        n += 1

    with open(src / "wav.scp", "w", encoding="utf-8") as f:
        for k, v in utt2wav.items():
            f.write(f"{k} {v}\n")
    with open(src / "text", "w", encoding="utf-8") as f:
        for k, v in utt2text.items():
            f.write(f"{k} {v}\n")
    with open(src / "utt2spk", "w", encoding="utf-8") as f:
        for k, v in utt2spk.items():
            f.write(f"{k} {v}\n")
    spks = sorted(set(utt2spk.values()))
    with open(src / "spk2utt", "w", encoding="utf-8") as f:
        for spk in spks:
            f.write(f"{spk} {' '.join(u for u in utt2wav if utt2spk[u] == spk)}\n")
    with open(src / "instruct", "w", encoding="utf-8") as f:  # CV3: instruct prefix per utt
        for k in utt2text:
            f.write(f"{k} {INSTRUCT_PREFIX}\n")
    return n


def _patched_train_config(repo: Path, work: Path, epochs: int, accum_grad: int,
                          lr: float | None, logline) -> Path:
    """Write a copy of the repo's cosyvoice3.yaml with our train_conf overrides.

    This version of train.py has no CLI config overrides (no --train_conf flag),
    so per-run settings go through a modified config copy in the run dir — the
    same flow upstream's fine-tune guide uses. The patch is scoped to the
    top-level ``train_conf:`` block only (never ``train_conf_gan:``).
    """
    src = repo / "examples" / "libritts" / "cosyvoice3" / "conf" / "cosyvoice3.yaml"
    dst = work / "cosyvoice3.train.yaml"
    lines = src.read_text(encoding="utf-8").split("\n")

    start = next((i for i, l in enumerate(lines) if re.match(r"^train_conf:\s*$", l)), None)
    if start is None:
        raise RuntimeError(f"train_conf section not found in {src} — upstream layout changed?")
    end = next((j for j in range(start + 1, len(lines)) if re.match(r"^\S", lines[j])), len(lines))
    block = "\n".join(lines[start:end])

    def sub(pattern: str, repl: str, what: str) -> None:
        nonlocal block
        block, n = re.subn(pattern, repl, block, count=1)
        if n == 0:
            logline(f"warning: could not set {what} in train_conf — upstream yaml may have changed")

    sub(r"max_epoch:\s*\S+", f"max_epoch: {epochs}", "max_epoch")
    sub(r"accum_grad:\s*\S+", f"accum_grad: {accum_grad}", "accum_grad")
    if lr is not None:
        # YAML 1.1 (PyYAML) only reads a float when the mantissa has a dot —
        # "4e-06" would parse as a string and break optim.Adam(lr=...).
        lrs = f"{lr:.6g}"
        if "e" in lrs and "." not in lrs.split("e")[0]:
            lrs = lrs.replace("e", ".0e", 1)
        sub(r"(optim_conf:\s*\n\s*)lr:\s*[\d.eE+-]+", rf"\g<1>lr: {lrs}", "optim_conf.lr")

    dst.write_text("\n".join(lines[:start]) + "\n" + block + "\n" + "\n".join(lines[end:]),
                   encoding="utf-8")
    return dst


def finetune(
    voice,
    ds: dict,
    report,
    logline,
    *,
    epochs: int,
    batch_size: int,  # accepted for interface parity; CV3 uses dynamic batching
    grad_accum_steps: int,
    precision: str,
    lr: float | None,
    force: bool,
) -> None:
    """Run the official CosyVoice 3 fine-tune recipe on the shared dataset.

    Trains the LLM component (where speaker identity lives), averages the best
    checkpoints, and builds an inference-ready model dir (symlink farm over the
    base weights + the trained llm.pt). Sets ``report.checkpoint`` to that dir.
    """
    eng = _engine()
    if not eng.is_installed():
        raise ExternalEngineError(
            f"Engine '{eng.name}' is not installed. Run: voiceclone install-engine {eng.name}"
        )
    repo = eng.engine_dir / "repo"
    # Self-heal the repo patches on every training run: finetune() only checks
    # the install marker (not ensure_installed), so a clone made before a patch
    # existed would otherwise never get repaired. Both are idempotent and touch
    # no network.
    _patch_repo_io(repo, logline)
    _patch_train_utils_join(repo, logline)
    model_dir = ensure_weights(logline)
    venv_py = str(eng.venv_python())

    settings = get_settings()
    ft_root = settings.models_dir / f"{voice.name}_ft_{eng.name}"
    run_name = time.strftime("run-%Y%m%d-%H%M%S")
    work = ft_root / run_name
    (work / "data").mkdir(parents=True, exist_ok=True)

    # 1. Kaldi-style data ------------------------------------------------------
    train_rows = _read_pipe_csv(Path(ds["train_csv"]))
    eval_rows = _read_pipe_csv(Path(ds["eval_csv"]))
    if not train_rows:
        raise RuntimeError("No training rows in the prepared dataset.")
    n_train = _build_split(work, "train", voice.name, Path(ds["out_dir"]), train_rows)
    # a validation split is required by their trainer; fall back to train data
    dev_rows = eval_rows if eval_rows else train_rows[: max(1, len(train_rows) // 10)]
    n_dev = _build_split(work, "dev", voice.name, Path(ds["out_dir"]), dev_rows)
    logline(f"CV3 data ready: {n_train} train / {n_dev} dev utterances under {work / 'data'}")

    # 2. parquet ----------------------------------------------------------------
    for split in ("train", "dev"):
        des = work / "data" / f"parquet_{split}"
        des.mkdir(parents=True, exist_ok=True)
        logline(f"Building parquet ({split}) ...")
        _run_streaming([
            venv_py, str(repo / "tools" / "make_parquet_list.py"),
            "--num_utts_per_parquet", "1000", "--num_processes", "4",
            "--src_dir", str(work / "data" / split),
            "--des_dir", str(des),
        ], logline, cwd=str(repo))
    (work / "data" / "train.data.list").write_text(
        Path(work / "data" / "parquet_train" / "data.list").read_text())
    (work / "data" / "dev.data.list").write_text(
        Path(work / "data" / "parquet_dev" / "data.list").read_text())

    # 3. train the LLM component -------------------------------------------------
    import torch

    cuda_ok = torch.cuda.is_available()
    exp_dir = work / "exp" / "llm" / "torch_ddp"
    # train.py has no CLI config overrides: bake epochs/accum_grad/lr into a
    # patched copy of the repo's yaml (see _patched_train_config).
    cfg = _patched_train_config(repo, work, epochs, max(1, grad_accum_steps), lr, logline)
    cmd = [
        str(eng.engine_dir / "venv" / "bin" / "torchrun"),
        "--nnodes=1", "--nproc_per_node=1",
        "--rdzv_id=1986", "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:29517",
        str(repo / "cosyvoice" / "bin" / "train.py"),
        "--train_engine", "torch_ddp",
        "--config", str(cfg),
        "--train_data", str(work / "data" / "train.data.list"),
        "--cv_data", str(work / "data" / "dev.data.list"),
        "--qwen_pretrain_path", str(model_dir / "CosyVoice-BlankEN"),
        "--onnx_path", str(model_dir),
        "--model", "llm",
        "--checkpoint", str(model_dir / "llm.pt"),
        "--model_dir", str(exp_dir),
        "--tensorboard_dir", str(work / "tensorboard" / "llm"),
        "--ddp.dist_backend", "nccl" if cuda_ok else "gloo",
        "--num_workers", "2", "--prefetch", "100", "--pin_memory",
    ]
    if cuda_ok:
        cmd.append("--use_amp")

    # torchrun's child process only gets the *script's* dir (cosyvoice/bin/) on
    # sys.path, and the repo is not pip-installed into the venv — without this,
    # train.py dies with "ModuleNotFoundError: No module named 'cosyvoice'".
    # Mirror what the inference worker does to its own sys.path.
    train_env = dict(os.environ)
    pp = train_env.get("PYTHONPATH", "")
    train_env["PYTHONPATH"] = os.pathsep.join(
        [str(repo), str(repo / "third_party" / "Matcha-TTS")] + ([pp] if pp else [])
    )

    logline(f"Training CV3 LLM: epochs={epochs} backend={'nccl' if cuda_ok else 'glo'} "
            f"({'GPU' if cuda_ok else 'CPU — expect this to be very slow'})")
    _run_streaming(cmd, logline, cwd=str(work), env=train_env)

    # 4. average the best checkpoints --------------------------------------------
    dst_llm = exp_dir / "llm.pt"
    logline("Averaging best checkpoints ...")
    _run_streaming([
        venv_py, str(repo / "cosyvoice" / "bin" / "average_model.py"),
        "--dst_model", str(dst_llm),
        "--src_path", str(exp_dir),
        "--num", "5", "--val_best",
    ], logline, cwd=str(repo))
    if not dst_llm.exists():
        raise RuntimeError(f"average_model did not produce {dst_llm}")

    # 5. build an inference-ready model dir (symlink farm + trained llm.pt) -------
    mdir = work / "model_dir"
    mdir.mkdir(parents=True, exist_ok=True)
    for entry in sorted(model_dir.iterdir()):
        if entry.name == "llm.pt":
            try:
                os.link(dst_llm, mdir / "llm.pt")  # hardlink: zero extra disk
            except OSError:
                shutil.copy2(dst_llm, mdir / "llm.pt")
        elif entry.name in ("README.md",):
            continue
        else:
            os.symlink(entry, mdir / entry.name)

    report.checkpoint = str(mdir)
    logline(f"Done. Inference model dir: {mdir}")
