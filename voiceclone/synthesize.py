"""High-level synthesis orchestration: emotion selection + engine dispatch."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from . import audio as A
from .config import get_settings
from .emotion import map_style_to_emotion, select_reference
from .engines.base import SynthesisResult
from .engines.xtts import LicenseNotAccepted, XttsEngine, is_license_accepted
from .voices import Voice


@dataclass
class SynthesisOutcome:
    result: SynthesisResult
    requested_emotion: str
    resolved_emotion: str
    output_path: Path | None = None


def synthesize(
    voice: Voice,
    text: str,
    emotion: str | None = None,
    style: str | None = None,
    language: str | None = None,
    engine_mode: str = "auto",  # auto | zero-shot | finetuned
    output_path: Path | None = None,
    engine: XttsEngine | None = None,
    temperature: float | None = None,
    length_penalty: float | None = None,
    repetition_penalty: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    speed: float | None = None,
    max_chars: int | None = None,  # long-text chunk cap (None = engine default 120, 0 = off)
) -> SynthesisOutcome:
    """Synthesize ``text`` with a registered voice.

    Steps:
      1. Resolve the requested emotion (preset or free-text style → preset).
      2. Pick the best reference sample for that emotion.
      3. Choose zero-shot vs fine-tuned mode.
      4. Run the engine, optionally save the WAV.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Nothing to synthesize (empty text).")

    # --- emotion resolution ------------------------------------------------
    requested = (emotion or "neutral").lower()
    if style:
        mapped = map_style_to_emotion(style)
        if mapped and requested in ("neutral", ""):
            requested = mapped
    resolved = requested if requested else "neutral"

    # --- reference selection ------------------------------------------------
    ref = select_reference(
        [s.to_dict() for s in voice.samples],
        resolved,
        get_settings().min_ref_seconds,
        get_settings().max_ref_seconds,
    )
    if ref is None:
        raise ValueError(f"Voice '{voice.name}' has no usable reference samples.")

    ref_path = voice.dir / ref.file
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference sample missing on disk: {ref_path}")

    # --- language -----------------------------------------------------------
    lang = (language or "").lower()
    if lang in ("auto", "", None):
        # Prefer the reference sample's language; fall back to most common.
        from collections import Counter

        ref_lang = _sample_lang(voice, ref.sample_id)
        counts = Counter(s.language for s in voice.samples)
        lang = ref_lang if ref_lang else (counts.most_common(1)[0][0] if counts else "en")

    # --- engine mode ---------------------------------------------------------
    finetuned_ckpt = None
    if engine_mode in ("auto", "finetuned") and voice.finetuned:
        ck = voice.finetuned.get("checkpoint")
        if ck and Path(ck).exists():
            finetuned_ckpt = ck
    if engine_mode == "finetuned" and finetuned_ckpt is None:
        raise ValueError(
            f"Voice '{voice.name}' has no fine-tuned checkpoint yet. Run `voiceclone train {voice.name}` first."
        )

    eng = engine or _default_engine()
    result = eng.synthesize(
        text=text,
        reference_wav_path=str(ref_path),
        reference_text=ref.transcript,
        language=lang,
        emotion=resolved,
        style=style,
        finetuned_checkpoint=finetuned_ckpt,
        temperature=temperature,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        speed=speed,
        max_chars=max_chars,
    )

    # --- save ----------------------------------------------------------------
    out: Path | None = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        A.save_wav(str(output_path), result.wav, result.sample_rate)
        out = output_path
    else:
        default_dir = get_settings().output_dir
        default_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe_text = "".join(c if c.isalnum() or c in " -" else "" for c in text[:24]).strip()[:24] or "clip"
        out = default_dir / f"{voice.name}_{stamp}_{safe_text}.wav"
        A.save_wav(str(out), result.wav, result.sample_rate)

    return SynthesisOutcome(
        result=result,
        requested_emotion=resolved,
        resolved_emotion=ref.emotion,
        output_path=out,
    )


def _sample_lang(voice: Voice, sample_id: str) -> str:
    for s in voice.samples:
        if s.id == sample_id:
            return s.language
    return "en"


def default_output_name(voice_name: str, text: str) -> Path:
    settings = get_settings()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_text = "".join(c if c.isalnum() or c in " -" else "" for c in text[:24]).strip()[:24] or "clip"
    return settings.output_dir / f"{voice_name}_{stamp}_{safe_text}.wav"


_engine_singleton: XttsEngine | None = None


def _default_engine() -> XttsEngine:
    global _engine_singleton
    if not is_license_accepted():
        raise LicenseNotAccepted()
    if _engine_singleton is None:
        from .engines.xtts import get_xtts_engine

        _engine_singleton = get_xtts_engine()
    return _engine_singleton
