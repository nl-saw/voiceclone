"""High-level synthesis orchestration: emotion selection + engine dispatch."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from . import audio as A
from .config import get_settings
from .emotion import ReferenceChoice, map_style_to_emotion, select_reference
from .engines.base import Engine, SynthesisResult
from .voices import Voice


@dataclass
class SynthesisOutcome:
    result: SynthesisResult
    requested_emotion: str
    resolved_emotion: str
    output_path: Path | None = None
    reference_source: str = "auto"  # auto (emotion-based pick) | explicit (user-chosen sample)


def synthesize(
    voice: Voice,
    text: str,
    emotion: str | None = None,
    style: str | None = None,
    language: str | None = None,
    engine_mode: str = "auto",  # auto | zero-shot | finetuned
    output_path: Path | None = None,
    engine_name: str | None = None,  # registry name; None → configured default
    engine: Engine | None = None,  # explicit instance (tests); wins over engine_name
    reference_sample: str | None = None,  # sample id or filename; skips auto-selection
    temperature: float | None = None,
    length_penalty: float | None = None,
    repetition_penalty: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    speed: float | None = None,
) -> SynthesisOutcome:
    """Synthesize ``text`` with a registered voice.

    Steps:
      1. Resolve the requested emotion (preset or free-text style → preset).
      2. Pick the reference sample — either the one the user named
         (``reference_sample``) or, when not given, the best match for the
         resolved emotion.
      3. Choose zero-shot vs fine-tuned mode.
      4. Run the engine, optionally save the WAV.

    Note: an explicit ``reference_sample`` always wins over the emotion
    selection — the emotion chips only steer the *automatic* pick.
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
    if reference_sample:
        ref = resolve_reference(voice, reference_sample)
        ref_source = "explicit"
    else:
        ref = select_reference(
            [s.to_dict() for s in voice.samples],
            resolved,
            get_settings().min_ref_seconds,
            get_settings().max_ref_seconds,
        )
        ref_source = "auto"
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

    # --- engine resolution ---------------------------------------------------
    eng = engine or get_cached_engine(engine_name)

    try:
        from .engines import EngineError, get_spec

        spec = get_spec(eng.name)
    except Exception:  # noqa: BLE001 — custom engine instance outside the registry
        spec = None

    # Language support check (only when we know the engine's language list).
    if spec is not None and lang not in spec.languages:
        raise ValueError(
            f"Engine '{spec.name}' does not support language '{lang}' "
            f"(supports: {', '.join(spec.languages)}). Pick a supported language or another "
            f"engine (see `voiceclone engines`)."
        )

    # --- engine mode ---------------------------------------------------------
    finetuned_ckpt = None
    ft_info = voice.finetuned_for(eng.name) if spec is not None else voice.finetuned
    ck = ft_info.get("checkpoint") if isinstance(ft_info, dict) else None
    if engine_mode in ("auto", "finetuned") and ck and Path(ck).exists():
        finetuned_ckpt = ck
    if engine_mode == "finetuned" and finetuned_ckpt is None:
        raise ValueError(
            f"Voice '{voice.name}' has no fine-tuned checkpoint for engine '{eng.name}'. "
            f"Run `voiceclone train {voice.name} --engine {eng.name}` first."
        )
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
        reference_source=ref_source,
    )


def resolve_reference(voice: Voice, ref_id: str) -> ReferenceChoice | None:
    """Resolve a user-named sample (by id like ``s002`` or by filename) to a
    reference choice. Returns None when the voice has no samples at all."""
    q = (ref_id or "").strip()
    if not q:
        raise ValueError("reference_sample is empty.")
    if not voice.samples:
        return None

    s = next((x for x in voice.samples if x.id == q), None)
    if s is None:  # try by filename (with or without extension, case-insensitive)
        ql = q.lower()
        s = next(
            (x for x in voice.samples
             if x.file.lower() == ql
             or Path(x.file).name.lower() == ql
             or Path(x.file).stem.lower() == ql),
            None,
        )
    if s is None:
        avail = ", ".join(x.id for x in voice.samples)
        raise ValueError(
            f"Sample '{q}' not found in voice '{voice.name}'. Available: {avail}"
        )

    if not (s.transcript or "").strip():
        raise ValueError(
            f"Sample '{s.id}' has no transcript, so it cannot be used as a "
            f"reference clip. Re-add it (or transcribe) to get one."
        )
    return ReferenceChoice(
        sample_id=s.id,
        file=s.file,
        transcript=s.transcript,
        emotion=(s.emotion or "neutral").lower(),
        duration_s=float(s.duration_s or 0.0),
        matched_requested_emotion=False,
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


# One engine instance per name: model loading is expensive, and engines cache
# their loaded weights internally keyed by checkpoint.
_engine_cache: dict[str, Engine] = {}


def get_cached_engine(name: str | None = None) -> Engine:
    from .engines import get_engine

    from .config import get_settings

    name = name or get_settings().default_engine
    if name not in _engine_cache:
        _engine_cache[name] = get_engine(name)
    return _engine_cache[name]
