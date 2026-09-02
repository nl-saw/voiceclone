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
from .voices import VoiceError, load_voice, normalize_finetuned, set_finetuned

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


def _run_info(run_dir: Path, registered_ckpt: str | None, engine: str = "xtts-v2",
              checkpoint_glob: str = "best_model_*.pth") -> dict:
    best_models = []
    for f in sorted(run_dir.glob(checkpoint_glob)):
        rel = str(f.relative_to(run_dir))  # keeps subdir prefixes (e.g. model_dir/llm.pt)
        best_models.append({
            "file": rel,
            "step": _step_of(f.name),
            "bytes": f.stat().st_size,
        })
    # also surface the plain best_model.pth copy (XTTS trainer duplicate of the best)
    dup = run_dir / "best_model.pth"
    has_dup = engine == "xtts-v2" and dup.exists()
    reg = bool(registered_ckpt and _is_within(Path(registered_ckpt), run_dir))
    return {
        "dir": run_dir.name,
        "path": str(run_dir),
        "engine": engine,
        "bytes": _dir_size(run_dir),
        "mtime": int(run_dir.stat().st_mtime),
        "best_models": best_models,
        "has_duplicate_best": has_dup,
        "registered": reg,
    }


def _engine_ft_roots(settings: Settings) -> list[tuple[str, str]]:
    """(engine_name, run_root_path) for every fine-tunable engine."""
    from .engines import REGISTRY

    out = []
    for name in sorted(REGISTRY):
        spec = REGISTRY[name]
        if not spec.finetune or not spec.finetune_root:
            continue
        root = settings.models_dir / spec.finetune_root.format(voice="{voice}") / spec.runs_subdir
        out.append((name, str(root)))
    return out


def _run_roots_for_voice(voice_name: str, settings: Settings) -> list[tuple[str, Path]]:
    out = []
    for name, tmpl in _engine_ft_roots(settings):
        out.append((name, Path(tmpl.format(voice=voice_name))))
    return out


def list_runs(voice_name: str, settings: Settings | None = None) -> dict:
    """All fine-tune run dirs for one voice (all engines), with sizes + registration."""
    from .engines import REGISTRY

    settings = settings or get_settings()
    v = load_voice(voice_name)  # raises VoiceError if unknown
    registered_by_engine = {
        eng: info.get("checkpoint")
        for eng, info in normalize_finetuned(v.finetuned).items()
    }
    runs = []
    for engine, ft_root in _run_roots_for_voice(v.name, settings):
        if not ft_root.exists():
            continue
        glob = REGISTRY[engine].checkpoint_glob
        for d in sorted(ft_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if d.is_dir():
                runs.append(_run_info(d, registered_by_engine.get(engine), engine=engine, checkpoint_glob=glob))
    return {
        "voice": v.name,
        "registered": registered_by_engine,
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

    from .engines import REGISTRY

    engine_suffixes = [
        (name, REGISTRY[name].finetune_root.split("{voice}")[1], REGISTRY[name])
        for name in sorted(REGISTRY)
        if REGISTRY[name].finetune and REGISTRY[name].finetune_root
    ]

    breakdown = [
        {"key": "base_model", "label": "Base model weights (required — do not delete)",
         "path": str(settings.models_dir),
         "bytes": size(settings.models_dir / "XTTS_v2_original_model_files")},
        {"key": "voices_samples", "label": "Voice source samples (your audio + transcripts)",
         "path": str(settings.voices_dir), "bytes": size(settings.voices_dir)},
    ]

    # per-voice fine-tune artifacts + traindata (all engines)
    ft_total = 0
    runs: list[dict] = []
    models_dir = settings.models_dir
    if models_dir.exists():
        for d in sorted(models_dir.iterdir()):
            if not d.is_dir():
                continue
            match = next(
                ((eng, suf, spec) for eng, suf, spec in engine_suffixes if d.name.endswith(suf)), None
            )
            if not match:
                continue
            engine, suf, spec = match
            voice = d.name[: -len(suf)]
            run_root = d / spec.runs_subdir
            vsize = size(d)
            ft_total += vsize
            breakdown.append({"key": f"ft:{voice}:{engine}", "label": f"Fine-tune runs — {voice} ({engine})",
                              "path": str(d), "bytes": vsize})
            if run_root.exists():
                for rd in sorted(run_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                    if not rd.is_dir():
                        continue
                    try:
                        v = load_voice(voice)
                        reg = (normalize_finetuned(v.finetuned).get(engine) or {}).get("checkpoint")
                    except VoiceError:
                        reg = None
                    info = _run_info(rd, reg, engine=engine, checkpoint_glob=spec.checkpoint_glob)
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

def _resolve_run(voice_name: str, run_dir_name: str | None, settings: Settings,
                 engine: str | None = None) -> tuple[str, Path]:
    """Resolve a run dir by name (or newest by mtime). Returns (engine, path)."""
    from .engines import get_spec

    if engine:
        spec = get_spec(engine)
        cands = [(engine, settings.models_dir / spec.finetune_root.format(voice=voice_name) / spec.runs_subdir)]
    else:
        cands = _run_roots_for_voice(voice_name, settings)

    all_runs: list[tuple[str, Path]] = []
    for eng, ft_root in cands:
        if ft_root.exists():
            all_runs.extend((eng, d) for d in ft_root.iterdir() if d.is_dir())
    if not all_runs:
        raise VoiceError(f"No fine-tune runs found for voice '{voice_name}'.")
    if run_dir_name:
        match = [(e, d) for e, d in all_runs if d.name == run_dir_name]
        if not match:
            raise VoiceError(f"Run '{run_dir_name}' not found for voice '{voice_name}'. "
                             f"Available: {', '.join(sorted(d.name for _, d in all_runs))}")
        return match[0]
    eng, newest = max(all_runs, key=lambda t: t[1].stat().st_mtime)
    return eng, newest


def _registered_run_dir(voice_name: str, settings: Settings) -> tuple[str, Path] | None:
    """The run dir holding ANY currently registered checkpoint (all engines)."""
    v = load_voice(voice_name)
    for eng, info in normalize_finetuned(v.finetuned).items():
        ck = info.get("checkpoint") if isinstance(info, dict) else None
        if not ck:
            continue
        ckp = Path(ck).resolve()
        for eng2, root in _run_roots_for_voice(voice_name, settings):
            if eng2 != eng or not root.exists():
                continue
            for rd in root.glob("*"):
                if rd.is_dir() and _is_within(ckp, rd):
                    return eng, rd
    return None


def clean_voice(
    voice_name: str,
    action: str,  # "run" | "all-but-registered" | "reset"
    run: str | None = None,
    force: bool = False,
    settings: Settings | None = None,
    engine: str | None = None,  # restrict cleanup to one engine (default: all)
) -> dict:
    """Delete fine-tune artifacts for a voice. Returns freed bytes + deleted paths.

    Safety: the base model and source samples are never touched. A run holding
    the currently registered checkpoint is protected unless you explicitly reset
    (which also clears the registration).
    """
    from .engines import get_spec

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

    reg = _registered_run_dir(voice_name, settings)  # (engine, dir) | None
    traindata = settings.models_dir / f"{voice_name}_traindata"

    if action == "run":
        eng, target = _resolve_run(voice_name, run, settings, engine=engine)
        if reg is not None and target.resolve() == reg[1].resolve():
            raise VoiceError(
                f"Refusing to delete '{target.name}' ({eng}): it holds the currently registered "
                f"checkpoint. Switch to another run or clear the registration first."
            )
        rm(target)

    elif action == "all-but-registered":
        if reg is None and not force:
            raise VoiceError(
                "No checkpoint is currently registered, so 'keep only registered' would delete "
                "ALL runs. Re-register a checkpoint first, or use action 'reset' to wipe everything."
            )
        if engine:
            spec = get_spec(engine)
            roots = [(engine, settings.models_dir / spec.finetune_root.format(voice=voice_name) / spec.runs_subdir)]
        else:
            roots = _run_roots_for_voice(voice_name, settings)
        for eng, ft_root in roots:
            if not ft_root.exists():
                continue
            for d in sorted(ft_root.iterdir()):
                if not d.is_dir():
                    continue
                if reg is not None and d.resolve() == reg[1].resolve():
                    # keep the registered run, but drop its duplicate best_model.pth copy (XTTS)
                    dup = d / "best_model.pth"
                    kept_best = [f for f in d.glob("best_model_*.pth")]
                    if eng == "xtts-v2" and dup.exists() and any(f.resolve() != dup.resolve() for f in kept_best):
                        rm(dup)
                    continue
                rm(d)

    elif action == "reset":
        # wipe the entire fine-tune tree (all engines, or one) + dataset, clear registration
        from .engines import REGISTRY

        if engine:
            specs = [get_spec(engine)]
        else:
            specs = [REGISTRY[n] for n in sorted(REGISTRY)
                     if REGISTRY[n].finetune and REGISTRY[n].finetune_root]
        for spec in specs:
            rm(settings.models_dir / spec.finetune_root.format(voice=voice_name))
        rm(traindata)
        from .voices import clear_all_finetuned

        clear_all_finetuned(voice_name)

    else:
        raise VoiceError(f"Unknown cleanup action '{action}'. Use 'run', 'all-but-registered', or 'reset'.")

    return {
        "voice": voice_name,
        "action": action,
        "freed_bytes": freed,
        "deleted": deleted,
        "registration_cleared": action == "reset",
    }


def register_checkpoint(voice_name: str, checkpoint: str, engine: str = "xtts-v2",
                        settings: Settings | None = None) -> dict:
    """Point a voice's fine-tuned registration (per engine) at an existing checkpoint."""
    from .engines import get_spec

    get_spec(engine)  # validate engine name
    v = load_voice(voice_name)
    ck = Path(checkpoint).expanduser().resolve()
    if not ck.exists():
        raise VoiceError(f"Checkpoint not found: {ck}")
    info = {
        "checkpoint": str(ck),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epochs": None,
        "n_train_sentences": len(v.samples),
    }
    set_finetuned(voice_name, info, engine=engine)
    return {"voice": v.name, "engine": engine, "registered": str(ck)}


def clear_checkpoint(voice_name: str, settings: Settings | None = None) -> dict:
    """Remove ALL fine-tuned registrations → synthesis falls back to zero-shot."""
    from .voices import clear_all_finetuned

    load_voice(voice_name)
    clear_all_finetuned(voice_name)
    return {"voice": voice_name, "registered": None}
