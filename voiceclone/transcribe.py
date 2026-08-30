"""Automatic transcription of voice samples with faster-whisper (CPU-friendly).

Used for two things:
  1. Storing a transcript with each registered sample (shown in the UI, used as
     prompt text for zero-shot synthesis).
  2. Preparing sentence-level training pairs for fine-tuning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_WHISPER_SR = 16000

_model_cache: dict[str, object] = {}


def get_whisper(model_size: str = "small", device: str = "auto"):
    """Lazily load (and cache) a faster-whisper model.

    Transcription runs on **CPU int8** by default (``"auto"`` → CPU). CTranslate2's
    GPU path needs the *system* cuBLAS 12, which is absent when CUDA comes from
    pip wheels — torch's CUDA-13 build bundles ``libcublas.so.13``, not ``.so.12`` —
    so guessing at CUDA (as we used to) failed at first inference with
    ``Library libcublas.so.12 is not found or cannot be loaded``. CPU int8 runs
    ~10x real-time, which is plenty for a one-off transcription step and can't be
    broken by ``uv sync`` (it's code, not an env/lib dependency).

    Pass ``device="cuda"`` to force the GPU path; it fails loudly if your
    CTranslate2 build can't load cuBLAS.
    """
    if model_size in _model_cache:
        return _model_cache[model_size]
    from faster_whisper import WhisperModel

    if device == "cuda":
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        # Force CTranslate2's CUDA executor to initialize now so a missing cuBLAS
        # surfaces here (loudly) instead of at the first real transcription.
        model.transcribe(np.zeros(1600, dtype=np.float32), beam_size=1)
    else:  # "auto" and "cpu" → CPU (robust default)
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    _model_cache[model_size] = model
    return model


@dataclass
class Transcription:
    text: str
    language: str
    language_probability: float


def transcribe_wav(
    wav: np.ndarray,
    sr: int,
    model_size: str = "small",
    language: str | None = None,
) -> Transcription:
    """Transcribe a mono float32 waveform.

    ``language`` may be an ISO code ("en", "nl") to force it, or None to let
    Whisper auto-detect (it is biased toward the toolkit's configured languages
    by simply running detection first).
    """
    from .audio import resample

    if language in ("auto", ""):
        language = None
    audio16 = resample(wav, sr, _WHISPER_SR)
    model = get_whisper(model_size)
    segments, info = model.transcribe(
        audio16,
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    parts: list[str] = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            parts.append(t)
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return Transcription(
        text=text,
        language=info.language,
        language_probability=float(info.language_probability),
    )


def split_sentences(text: str) -> list[str]:
    """Split transcript text into sentence-like chunks for training pairs.

    Splits on terminal punctuation and newlines; merges fragments that are too
    short so the training set contains usable sentences.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    raw = re.split(r"(?<=[.!?…])\s+|\n+", text)
    chunks: list[str] = []
    buf = ""
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        buf = f"{buf} {piece}".strip()
        if len(piece.split()) >= 4 or re.search(r"[.!?…]$", piece):
            chunks.append(buf)
            buf = ""
    if buf:
        chunks.append(buf)
    # Drop tiny fragments (XTTS wants meaningful prompt texts).
    return [c for c in chunks if len(c.split()) >= 3]
