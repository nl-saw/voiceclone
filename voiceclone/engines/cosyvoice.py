"""CosyVoice 3 engine (FunAudioLLM) — zero-shot cloning + official fine-tuning.

* Apache-2.0 code AND weights; actively maintained upstream
  (repo: https://github.com/FunAudioLLM/CosyVoice, model: Fun-CosyVoice3-0.5B-2512).
* 9 languages for zero-shot cloning: zh/en/ja/ko/de/es/fr/it/ru — **no Dutch**.
* Runs in a dedicated venv (upstream hard-pins torch==2.3.1, transformers==4.51.3,
  Python 3.10) that conflicts with the toolkit's own dependencies, so it is driven
  through the external-engine worker protocol (see voiceclone/engines/external.py).

Install:   voiceclone install-engine cosyvoice3
Fine-tune: voiceclone train <voice> --engine cosyvoice3
           (official recipe: Kaldi-style data → parquet → torchrun train.py on the
           LLM component → averaged checkpoint → symlink-farm model dir)
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from ..config import get_settings
from .base import SynthesisResult  # noqa: F401 — re-exported for typing
from .external import ExternalEngine, ExternalEngineError

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


def ensure_installed(logline=print) -> None:
    """Clone the repo, create the dedicated venv, install dependencies.

    Idempotent; safe to re-run (reuses existing repo/venv). Weights are NOT
    downloaded here — call :func:`ensure_weights` (or ``init --download``).
    """
    eng = _engine()
    engine_dir = eng.engine_dir
    engine_dir.mkdir(parents=True, exist_ok=True)
    if (engine_dir / "installed.json").exists():
        logline("Already installed.")
        return

    # 1. repo -----------------------------------------------------------------
    repo = engine_dir / "repo"
    if not (repo / ".git").exists():
        logline(f"Cloning {REPO_URL} → {repo}")
        _run_streaming(
            ["git", "clone", "--recursive", REPO_URL, str(repo)], logline
        )
    elif not (repo / "third_party" / "Matcha-TTS").exists():
        logline("Updating git submodules (Matcha-TTS) ...")
        _run_streaming(["git", "-C", str(repo), "submodule", "update", "--init", "--recursive"], logline)

    # 2. venv -----------------------------------------------------------------
    venv_py = eng.venv_python()
    uv = shutil.which("uv")
    reqs = repo / "requirements.txt"
    if not reqs.exists():
        raise RuntimeError(f"{reqs} not found — is the repo clone complete?")
    if not venv_py.exists():
        if uv:
            logline(f"Creating Python {PYTHON_VERSION} venv with uv (interpreter may be downloaded) ...")
            _run_streaming(
                [uv, "venv", "--python", PYTHON_VERSION, str(engine_dir / "venv")], logline
            )
        else:
            py = shutil.which(f"python{PYTHON_VERSION}") or shutil.which("python3.10")
            if not py:
                raise RuntimeError(
                    f"Need Python {PYTHON_VERSION} for CosyVoice (or 'uv' to fetch it). "
                    "Install uv: https://docs.astral.sh/uv/"
                )
            logline(f"Creating venv with {py} ...")
            _run_streaming([py, "-m", "venv", str(engine_dir / "venv")], logline)

    # 3. dependencies (idempotent; fast no-op when already satisfied) ----------
    logline("Installing CosyVoice dependencies (torch 2.3.1, transformers 4.51.3, ...) — this takes a while")
    # deepspeed ships sdist-only on PyPI. Its setup.py probes the CUDA toolkit
    # (nvcc) whenever torch sees a GPU at *build* time and hard-fails with
    # MissingCUDAException otherwise — so on a GPU box without the CUDA
    # toolkit the install breaks. DS_ACCELERATOR=cpu makes setup.py build
    # deepspeed as pure Python (no compiled ops); that is all we need, since
    # CosyVoice fine-tuning runs with --train_engine torch_ddp and only uses
    # deepspeed for its Python-side utilities. Build-time env only — the
    # worker process never sees it.
    install_env = {**os.environ, "DS_ACCELERATOR": "cpu"}
    if uv:
        # upstream requirements.txt pulls from official vendor indexes
        # (pytorch cu121, onnxruntime-cuda-12); uv needs the best-match
        # strategy to resolve across them. A few sdists (openai-whisper,
        # pyworld, deepspeed) import build-time deps that are not declared →
        # build with the venv's own environment (--no-build-isolation), which
        # means pre-installing setuptools<81 (newer dropped pkg_resources),
        # numpy and torch first.
        _run_streaming([uv, "pip", "install", "--python", str(venv_py), "setuptools<81", "wheel"], logline)
        _run_streaming([
            uv, "pip", "install", "--python", str(venv_py),
            "torch==2.3.1", "torchaudio==2.3.1", "numpy==1.26.4",
            "--index-strategy", "unsafe-best-match",
        ], logline, env=install_env)
        _run_streaming([
            uv, "pip", "install", "--python", str(venv_py), "-r", str(reqs),
            "--index-strategy", "unsafe-best-match", "--no-build-isolation",
        ], logline, env=install_env)
    else:
        _run_streaming([str(venv_py), "-m", "pip", "install", "-U", "pip"], logline)
        _run_streaming([str(venv_py), "-m", "pip", "install", "-r", str(reqs)], logline, env=install_env)

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
    _run_streaming([str(eng.venv_python()), "-c", code], logline)
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
    cmd = [
        str(eng.engine_dir / "venv" / "bin" / "torchrun"),
        "--nnodes=1", "--nproc_per_node=1",
        "--rdzv_id=1986", "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:29517",
        str(repo / "cosyvoice" / "bin" / "train.py"),
        "--train_engine", "torch_ddp",
        "--config", str(repo / "examples" / "libritts" / "cosyvoice3" / "conf" / "cosyvoice3.yaml"),
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
        "--train_conf.max_epoch", str(epochs),
        "--train_conf.accum_grad", str(max(1, grad_accum_steps)),
    ]
    if cuda_ok:
        cmd.append("--use_amp")
    if lr:
        cmd += ["--train_conf.optim_conf.lr", str(lr)]

    logline(f"Training CV3 LLM: epochs={epochs} backend={'nccl' if cuda_ok else 'glo'} "
            f"({'GPU' if cuda_ok else 'CPU — expect this to be very slow'})")
    _run_streaming(cmd, logline, cwd=str(work))

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
