"""Disk-usage inspection and safe cleanup for fine-tuning artifacts.

Fine-tuning writes large checkpoints (~5.6 GB each) under
``data/models/<voice>_ft/run/training/<run>/`` — every run keeps a
``best_model.pth`` copy *and* a ``best_model_<step>.pth``, so a few experiments
quickly consume tens of gigabytes. This module lets the web UI (and scripts)
see what is taking space and delete runs safely, without ever touching:

  * the base XTTS v2 weights (``data/models/XTTS_v2_original_model_files/``),
  * the voice's source samples (``data/voices/<name>/samples/``),
  * a run that still holds the *currently registered* checkpoint
    (you must switch or clear the registration first).

All paths are derived from ``Settings`` so this is move-safe.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

from .config import Settings, get_settings
from .voices import VoiceError, load_voice, set_finetuned

# Below this much source audio, fine-tuning an XTTS v2 GPT tends to *hurt*
# word accuracy rather than help (it perturbs the pretrained text->speech
# mapping faster than it learns voice-specific traits). Advisory threshold.
RECOMMENDED_MIN_AUDIO_S = 600  # ~10 minutes


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def _step_of(name: str) -> int:
    m = re.search(r"(\d+)\.pth$", name)
    return int(m.group(1)) if m else -1


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _run_info(run_dir: Path, registered_ckpt: str | None) -> dict:
    best_models = []
    for f in sorted(run_dir.glob("best_model_*.pth")):
        best_models.append({
            "file": f.name,
            "step": _step_of(f.name),
            "bytes": f.stat().st_size,
        })
    # also surface the plain best_model.pth copy (duplicate of the best)
    dup = run_dir / "best_model.pth"
    has_dup = dup.exists()
    reg = bool(registered_ckpt and _is_within(Path(registered_ckpt), run_dir))
    return {
        "dir": run_dir.name,
        "path": str(run_dir),
        "bytes": _dir_size(run_dir),
        "mtime": int(run_dir.stat().st_mtime),
        "best_models": best_models,
        "has_duplicate_best": has_dup,
        "registered": reg,
    }


def list_runs(voice_name: str, settings: Settings | None = None) -> dict:
    """All fine-tune run dirs for one voice, with sizes + registration state."""
    settings = settings or get_settings()
    v = load_voice(voice_name)  # raises VoiceError if unknown
    registered = (v.finetuned or {}).get("checkpoint")
    ft_root = settings.models_dir / f"{v.name}_ft" / "run" / "training"
    runs = []
    if ft_root.exists():
        for d in sorted(ft_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if d.is_dir():
                runs.append(_run_info(d, registered))
    return {
        "voice": v.name,
        "registered": registered,
        "runs": runs,
        "traindata_bytes": _dir_size(settings.models_dir / f"{v.name}_traindata")
        if (settings.models_dir / f"{v.name}_traindata").exists() else 0,
    }


def scan_storage(settings: Settings | None = None) -> dict:
    """Break down where disk space is going under the data dir."""
    settings = settings or get_settings()
    dd = settings.data_dir

    def size(p: Path) -> int:
        return _dir_size(p) if p.exists() else 0

    breakdown = [
        {"key": "base_model", "label": "Base XTTS v2 model (required — do not delete)",
         "path": str(settings.models_dir / "XTTS_v2_original_model_files"),
         "bytes": size(settings.models_dir / "XTTS_v2_original_model_files")},
        {"key": "voices_samples", "label": "Voice source samples (your audio + transcripts)",
         "path": str(settings.voices_dir), "bytes": size(settings.voices_dir)},
    ]

    # per-voice fine-tune artifacts + traindata
    ft_total = 0
    runs: list[dict] = []
    models_dir = settings.models_dir
    if models_dir.exists():
        for d in sorted(models_dir.iterdir()):
            if not d.is_dir() or not d.name.endswith("_ft"):
                continue
            voice = d.name[: -len("_ft")]
            run_root = d / "run" / "training"
            vsize = size(d)
            ft_total += vsize
            breakdown.append({"key": f"ft:{voice}", "label": f"Fine-tune runs — {voice}",
                              "path": str(d), "bytes": vsize})
            if run_root.exists():
                for rd in sorted(run_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                    if not rd.is_dir():
                        continue
                    try:
                        v = load_voice(voice)
                        reg = (v.finetuned or {}).get("checkpoint")
                    except VoiceError:
                        reg = None
                    info = _run_info(rd, reg)
                    info["voice"] = voice
                    runs.append(info)
            td = models_dir / f"{voice}_traindata"
            if td.exists():
                breakdown.append({"key": f"traindata:{voice}", "label": f"Training dataset — {voice}",
                                  "path": str(td), "bytes": size(td)})

    for key, label, p in [
        ("logs", "Training logs", dd / "logs"),
        ("output", "Synthesized output (generated speech)", dd / "output"),
        ("cache", "Cache (Whisper models etc.)", dd / "cache"),
    ]:
        b = size(p)
        if b or p.exists():
            breakdown.append({"key": key, "label": label, "path": str(p), "bytes": b})

    total = sum(x["bytes"] for x in breakdown)
    return {
        "data_dir": str(dd),
        "total_bytes": total,
        "ft_total_bytes": ft_total,
        "recommended_min_audio_s": RECOMMENDED_MIN_AUDIO_S,
        "breakdown": sorted(breakdown, key=lambda x: -x["bytes"]),
        "runs": runs,
    }


# --------------------------------------------------------------------------- #
# Cleanup actions
# --------------------------------------------------------------------------- #

def _resolve_run(voice_name: str, run_dir_name: str | None, settings: Settings) -> Path:
    ft_root = settings.models_dir / f"{voice_name}_ft" / "run" / "training"
    if not ft_root.exists():
        raise VoiceError(f"No fine-tune runs found for voice '{voice_name}'.")
    cands = [d for d in ft_root.iterdir() if d.is_dir()]
    if run_dir_name:
        match = [d for d in cands if d.name == run_dir_name]
        if not match:
            raise VoiceError(f"Run '{run_dir_name}' not found for voice '{voice_name}'. "
                             f"Available: {', '.join(sorted(d.name for d in cands))}")
        return match[0]
    # no explicit name → the newest run by mtime
    if not cands:
        raise VoiceError(f"No fine-tune runs found for voice '{voice_name}'.")
    return max(cands, key=lambda p: p.stat().st_mtime)


def _registered_run_dir(voice_name: str, settings: Settings) -> Path | None:
    v = load_voice(voice_name)
    ck = (v.finetuned or {}).get("checkpoint")
    if not ck:
        return None
    ckp = Path(ck).resolve()
    ft_root = settings.models_dir / f"{voice_name}_ft" / "run" / "training"
    for d in ft_root.glob("*"):
        if d.is_dir() and _is_within(ckp, d):
            return d
    return None


def clean_voice(
    voice_name: str,
    action: str,  # "run" | "all-but-registered" | "reset"
    run: str | None = None,
    force: bool = False,
    settings: Settings | None = None,
) -> dict:
    """Delete fine-tune artifacts for a voice. Returns freed bytes + deleted paths.

    Safety: the base model and source samples are never touched. A run holding
    the currently registered checkpoint is protected unless you explicitly reset
    (which also clears the registration).
    """
    settings = settings or get_settings()
    load_voice(voice_name)  # validate voice exists
    deleted: list[str] = []
    freed = 0

    def rm(path: Path) -> None:
        nonlocal freed
        if path.exists():
            b = _dir_size(path) if path.is_dir() else path.stat().st_size
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            freed += b
            deleted.append(str(path))

    reg_dir = _registered_run_dir(voice_name, settings)
    ft_root = settings.models_dir / f"{voice_name}_ft" / "run" / "training"
    traindata = settings.models_dir / f"{voice_name}_traindata"

    if action == "run":
        target = _resolve_run(voice_name, run, settings)
        if reg_dir is not None and target.resolve() == reg_dir.resolve():
            raise VoiceError(
                f"Refusing to delete '{target.name}': it holds the currently registered "
                f"checkpoint. Switch to another run or clear the registration first."
            )
        rm(target)

    elif action == "all-but-registered":
        if reg_dir is None and not force:
            raise VoiceError(
                "No checkpoint is currently registered, so 'keep only registered' would delete "
                "ALL runs. Re-register a checkpoint first, or use action 'reset' to wipe everything."
            )
        for d in sorted(ft_root.iterdir()) if ft_root.exists() else []:
            if not d.is_dir():
                continue
            if reg_dir is not None and d.resolve() == reg_dir.resolve():
                # keep the registered run, but drop its duplicate best_model.pth copy
                dup = d / "best_model.pth"
                kept_best = [f for f in d.glob("best_model_*.pth")]
                if dup.exists() and any(f.resolve() != dup.resolve() for f in kept_best):
                    rm(dup)
                continue
            rm(d)

    elif action == "reset":
        # wipe the entire fine-tune tree + the prepared dataset, clear registration
        ft_root_parent = settings.models_dir / f"{voice_name}_ft"
        rm(ft_root_parent)
        rm(traindata)
        set_finetuned(voice_name, None)

    else:
        raise VoiceError(f"Unknown cleanup action '{action}'. Use 'run', 'all-but-registered', or 'reset'.")

    return {
        "voice": voice_name,
        "action": action,
        "freed_bytes": freed,
        "deleted": deleted,
        "registration_cleared": action == "reset",
    }


def register_checkpoint(voice_name: str, checkpoint: str, settings: Settings | None = None) -> dict:
    """Point a voice's fine-tuned registration at an existing checkpoint file."""
    settings = settings or get_settings()
    v = load_voice(voice_name)
    ck = Path(checkpoint).expanduser().resolve()
    if not ck.exists():
        raise VoiceError(f"Checkpoint not found: {ck}")
    # infer epochs from the step number when possible (65 steps/epoch here)
    info = {
        "checkpoint": str(ck),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epochs": None,
        "n_train_sentences": len(v.samples),
    }
    set_finetuned(voice_name, info)
    return {"voice": v.name, "registered": str(ck)}


def clear_checkpoint(voice_name: str, settings: Settings | None = None) -> dict:
    """Remove the fine-tuned registration → synthesis falls back to zero-shot."""
    load_voice(voice_name)
    set_finetuned(voice_name, None)
    return {"voice": voice_name, "registered": None}
