"""Fine-tuning pipeline for per-voice XTTS v2 models.

Follows the official coqui fine-tuning recipe (XTTS GPT trainer):

  1. Prepare a dataset from the voice's registered samples:
     sentence-level WAV clips + pipe-delimited CSV (audio_file|text|speaker_name),
     split into train/eval, one dataset config per language (bilingual voices OK).
  2. Run the official GPTTrainer starting from the original XTTS v2 weights.
  3. Record the resulting checkpoint in the voice profile so synthesis can use it.

CPU note: fine-tuning on CPU works but is slow (expect hours for a few minutes
of audio). On a GPU machine this takes minutes-to-an-hour. Use --dry-run to
prepare data without training.
"""

from __future__ import annotations

import contextlib
import csv
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_settings
from .voices import Voice, VoiceError


@dataclass
class TrainReport:
    voice: str
    n_train: int = 0
    n_eval: int = 0
    total_audio_seconds: float = 0.0
    languages: list[str] = field(default_factory=list)
    checkpoint: str | None = None
    output_dir: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None


def prepare_dataset(voice: Voice, out_dir: Path) -> dict:
    """Build the training dataset directory from registered samples.

    Layout (what the coqui "coqui" formatter expects):
      <out>/wavs/<id>_<sentence>.wav
      <out>/metadata_train.csv   (pipe-delimited: audio_file|text|speaker_name)
      <out>/metadata_eval.csv
    """
    from . import audio as A
    from .transcribe import split_sentences

    out_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = out_dir / "wavs"
    if wavs_dir.exists():
        shutil.rmtree(wavs_dir)
    wavs_dir.mkdir()

    rows: list[dict] = []
    for s in voice.samples:
        text = (s.transcript or "").strip()
        if not text:
            continue
        wav_path = voice.dir / s.file
        if not wav_path.exists():
            continue
        sr = get_settings().sample_rate
        wav = A.load_audio(str(wav_path), sr)
        sentences = split_sentences(text)
        if not sentences:
            sentences = [text]

        # Proportionally slice the audio across sentences by word count.
        total_words = max(1, sum(len(x.split()) for x in sentences))
        pos = 0
        for i, sent in enumerate(sentences):
            frac = len(sent.split()) / total_words
            start = int(pos * len(wav))
            end = int((pos + frac) * len(wav)) if i < len(sentences) - 1 else len(wav)
            pos += frac
            clip = wav[start:end]
            if len(clip) < sr // 3:  # skip sub-0.33 s slivers
                continue
            name = f"{s.id}_{i:04d}.wav"
            A.save_wav(str(wavs_dir / name), clip, sr)
            rows.append({"audio_file": f"wavs/{name}", "text": sent.strip(), "speaker_name": voice.name})

    if not rows:
        raise RuntimeError(
            f"Voice '{voice.name}' has no usable samples (need audio with transcripts). "
            "Add samples first: voiceclone add-sample <voice> file.wav ..."
        )

    # 15% eval split, mirroring the official demo.
    import random

    random.seed(42)
    shuffled = rows[:]
    random.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * 0.15)) if len(shuffled) > 3 else 0
    eval_rows = sorted(shuffled[:n_eval], key=lambda r: r["audio_file"])
    train_rows = sorted(shuffled[n_eval:], key=lambda r: r["audio_file"])

    def write_csv(rows: list[dict], path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="|")
            w.writerow(["audio_file", "text", "speaker_name"])
            for r in rows:
                w.writerow([r["audio_file"], r["text"], r["speaker_name"]])

    write_csv(train_rows, out_dir / "metadata_train.csv")
    write_csv(eval_rows, out_dir / "metadata_eval.csv")

    langs = sorted({s.language for s in voice.samples if (s.transcript or "").strip()})
    return {
        "out_dir": str(out_dir),
        "train_csv": str(out_dir / "metadata_train.csv"),
        "eval_csv": str(out_dir / "metadata_eval.csv"),
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "languages": langs,
    }


def _find_latest_checkpoint(output_path: Path) -> Path | None:
    """Locate the newest GPT checkpoint produced by the trainer.

    The coqui trainer writes ``best_model_<step>.pth`` (plus a ``best_model.pth``
    copy of the current best) and ``checkpoint_<step>.pth`` snapshots — prefer
    the highest-step best model, then the newest checkpoint snapshot.
    """
    if not output_path.exists():
        return None

    def _step(p: Path) -> int:
        m = re.search(r"(\d+)\.pth$", p.name)
        return int(m.group(1)) if m else -1

    def _newest(cands: list[Path]) -> Path:
        # A later run has FEWER total steps (fewer epochs), so global max-step is
        # wrong once several runs exist. Pick the newest run directory first,
        # then the highest step inside it.
        runs = {p.parent for p in cands}
        if len(runs) > 1:
            newest = max(runs, key=lambda d: max(f.stat().st_mtime for f in d.iterdir()))
            cands = [p for p in cands if p.parent == newest]
        return max(cands, key=_step)

    best_models = list(output_path.rglob("best_model_*.pth"))
    if best_models:
        return _newest(best_models)
    checkpoints = list(output_path.rglob("checkpoint_*.pth"))
    if checkpoints:
        return _newest(checkpoints)
    fallback = list(output_path.rglob("*.pth")) + list(output_path.rglob("*.ckpt"))
    if not fallback:
        return None
    return max(fallback, key=lambda p: p.stat().st_mtime)


@contextlib.contextmanager
def _tee_stdout(log):
    """Mirror sys.stdout into the train log file while active.

    coqui's ``Trainer.fit()`` swallows training errors: it prints them via
    ``traceback.print_exc()`` (stdout) and then calls ``sys.exit(1)``, so the
    real exception (e.g. CUDA OOM) only reaches us as a bare SystemExit unless
    we capture what it printed.
    """
    if log is None:
        yield
        return

    class _Tee:
        def __init__(self, primary, secondary):
            self.primary = primary
            self.secondary = secondary

        def write(self, s):
            self.primary.write(s)
            try:
                self.secondary.write(s)
                self.secondary.flush()
            except Exception:  # noqa: BLE001 — logging must never break training
                pass

        def flush(self):
            self.primary.flush()

    orig = sys.stdout
    sys.stdout = _Tee(orig, log)
    try:
        yield
    finally:
        sys.stdout = orig


def _available_ram_gib() -> float | None:
    try:
        info = Path("/proc/meminfo").read_text().splitlines()
        for line in info:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return None


def run_finetune(
    voice_name: str,
    epochs: int = 5,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    dry_run: bool = False,
    force: bool = False,
    precision: str = "auto",  # auto | bf16 | fp32
    lr: float | None = None,  # default 4e-06 (lower = stays closer to the base voice)
    log_path: Path | None = None,
) -> TrainReport:
    """Fine-tune XTTS v2 on a voice's samples. Blocks until done (or error)."""
    from .voices import load_voice

    settings = get_settings()
    voice = load_voice(voice_name)
    report = TrainReport(voice=voice.name)

    log = open(log_path, "a", encoding="utf-8") if log_path else None

    def logline(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if log:
            log.write(line + "\n")
            log.flush()

    try:
        logline(f"Preparing dataset for voice '{voice.name}' ...")
        ds = prepare_dataset(voice, settings.models_dir / f"{voice.name}_traindata")
        report.n_train = ds["n_train"]
        report.n_eval = ds["n_eval"]
        report.languages = ds["languages"]
        report.output_dir = ds["out_dir"]
        logline(f"Dataset ready: {ds['n_train']} train / {ds['n_eval']} eval sentences, languages={ds['languages']}")

        # ---- precision selection -------------------------------------------
        # bf16 mixed precision on CUDA (master weights/optimizer stay fp32):
        # ~2x faster and lower activation memory. Plain fp32 otherwise.
        import torch

        cuda_ok = torch.cuda.is_available()
        if precision == "auto":
            use_bf16 = cuda_ok
        elif precision == "bf16":
            if not cuda_ok:
                raise VoiceError("--precision bf16 requested but no CUDA GPU is available on this machine.")
            use_bf16 = True
        else:  # fp32
            use_bf16 = False

        if use_bf16:
            logline(f"GPU detected: {torch.cuda.get_device_name(0)} — training with bfloat16 mixed precision")
        elif cuda_ok:
            logline("CUDA available but fp32 requested — full float32 training on GPU")
        else:
            logline("No CUDA GPU — training in float32 on CPU (slow)")

        if dry_run:
            logline("dry-run: skipping training.")
            return report

        # ---- pre-flight: RAM guard (fp32 518M params + AdamW ≈ 9-10 GB) ----
        avail = _available_ram_gib()
        MIN_RAM_GIB = 12.0
        if avail is not None and avail < MIN_RAM_GIB and not force:
            raise VoiceError(
                f"Not enough free RAM to fine-tune safely: {avail:.1f} GiB available, "
                f"~{MIN_RAM_GIB:.0f} GiB needed (model + gradients + optimizer states in float32). "
                "Starting would likely OOM-kill the process and hang the machine. "
                "Options: free up memory, run on a GPU/larger-RAM machine, or pass --force to start anyway."
            )

        # ---- official fine-tuning path ------------------------------------
        import os

        os.environ.setdefault("TRAINER_TELEMETRY", "0")  # disable coqui scarf.sh ping (also blocked on some networks)
        from .compat import install_audio_loader

        install_audio_loader()  # must precede trainer dataset imports (PyAV backend)
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.layers.xtts.trainer.gpt_trainer import (
            GPTArgs,
            GPTTrainer,
            GPTTrainerConfig,
            XttsAudioConfig,
        )
        from trainer import Trainer, TrainerArgs

        out_path = settings.models_dir / f"{voice.name}_ft"
        run_training = out_path / "run" / "training"
        os.makedirs(run_training, exist_ok=True)

        logline("Ensuring original XTTS v2 weights are cached ...")
        from .engines.xtts import ensure_weights

        # Downloads into data/models/XTTS_v2_original_model_files; reuse it.
        base = ensure_weights(allow_download=True)

        dvae = base["dvae"]
        mel = base["mel"]
        vocab = base["vocab"]
        xtts_ckpt = base["model"]
        xtts_config = base["config"]

        # One dataset config per language (bilingual voices are fine).
        ds_configs = [
            BaseDatasetConfig(
                formatter="coqui",
                dataset_name=f"ft_{voice.name}_{lang}",
                path=ds["out_dir"],
                meta_file_train=ds["train_csv"],
                meta_file_val=ds["eval_csv"],
                language=lang,
            )
            for lang in (ds["languages"] or ["en"])
        ]

        model_args = GPTArgs(
            max_conditioning_length=132300,  # ~6 s
            min_conditioning_length=66150,   # ~3 s
            debug_loading_failures=False,
            max_wav_length=255995,           # ~11.6 s
            max_text_length=200,
            mel_norm_file=mel,
            dvae_checkpoint=dvae,
            xtts_checkpoint=xtts_ckpt,
            tokenizer_file=vocab,
            gpt_num_audio_tokens=1026,
            gpt_start_audio_token=1024,
            gpt_stop_audio_token=1025,
            gpt_use_masking_gt_prompt_approach=True,
            gpt_use_perceiver_resampler=True,
        )
        audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
        config = GPTTrainerConfig(
            epochs=epochs,
            output_path=str(run_training),
            model_args=model_args,
            run_name=f"FT_{voice.name}",
            project_name="voiceclone",
            run_description=f"Fine-tune XTTS v2 on voice '{voice.name}'",
            dashboard_logger="tensorboard",  # value only; a no-op logger is passed to Trainer below
            logger_uri=None,
            audio=audio_config,
            batch_size=batch_size,
            batch_group_size=48,
            eval_batch_size=batch_size,
            # trainer: mixed_precision + precision="bf16" → torch.autocast(bf16) on CUDA,
            # no GradScaler (only fp16 uses one). Ignored entirely when mixed_precision=False.
            mixed_precision=use_bf16,
            precision="bf16" if use_bf16 else "fp16",  # fp16 = config default; unused unless mixed_precision
            num_loader_workers=0,  # in-process loading: safer on CPU-only boxes (no forked FFmpeg workers)
            eval_split_max_size=256,
            print_step=10,
            plot_step=100,
            log_model_step=100,
            save_step=1000,
            save_n_checkpoints=1,
            save_checkpoints=True,
            print_eval=False,
            optimizer="AdamW",
            optimizer_wd_only_on_weights=True,
            optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
            lr=lr if lr is not None else 4e-06,
            lr_scheduler="MultiStepLR",
            lr_scheduler_params={
                "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
                "gamma": 0.5,
                "last_epoch": -1,
            },
            test_sentences=[],
        )

        from .compat import legacy_torch_load

        with legacy_torch_load():  # torch>=2.6 weights_only breaks old checkpoints
            model = GPTTrainer.init_from_config(config)
        logline("Loading training samples ...")
        train_samples, eval_samples = load_tts_samples(
            ds_configs,
            eval_split=True,
            eval_split_max_size=config.eval_split_max_size,
            eval_split_size=config.eval_split_size,
        )
        logline(f"Starting trainer: epochs={epochs} batch_size={batch_size} grad_accum={grad_accum_steps}")
        from trainer.logging.dummy_logger import DummyLogger

        with legacy_torch_load(), _tee_stdout(log):  # tee stdout so the trainer's failure traceback lands in our log
            trainer = Trainer(
                TrainerArgs(
                    restore_path=None,
                    skip_train_epoch=False,
                    start_with_eval=False,
                    grad_accum_steps=grad_accum_steps,
                ),
                config,
                output_path=str(run_training),
                model=model,
                train_samples=train_samples,
                eval_samples=eval_samples,
                dashboard_logger=DummyLogger(),  # no wandb/tensorboard dependency
                parse_command_line_args=False,  # we own the CLI; don't let trainer touch sys.argv
            )
            trainer.fit()

        ckpt = _find_latest_checkpoint(run_training)
        if ckpt is None:
            raise RuntimeError(f"Trainer finished but no checkpoint found under {run_training}")
        report.checkpoint = str(ckpt)
        logline(f"Done. Checkpoint: {ckpt}")
        return report

    except BaseException as e:  # noqa: BLE001 — surface any failure in the report (incl. SystemExit from trainer.fit)
        if isinstance(e, SystemExit):
            report.error = "Trainer aborted with an error — see traceback above"
        else:
            report.error = f"{type(e).__name__}: {e}"
        if log:
            log.write(f"[{time.strftime('%H:%M:%S')}] FAILED: {report.error}\n")
        raise
    finally:
        if log:
            log.close()


def record_finetune(voice_name: str, report: TrainReport) -> None:
    """Store the finetuned checkpoint reference in the voice profile."""
    from .voices import set_finetuned

    if report.checkpoint:
        set_finetuned(
            voice_name,
            {
                "checkpoint": report.checkpoint,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(report.finished_at or time.time())),
                "epochs": None,
                "n_train_sentences": report.n_train,
            },
        )
