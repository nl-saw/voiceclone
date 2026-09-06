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
    # Word-level timestamps [(word, start_s, end_s), ...] on the ORIGINAL audio
    # timeline (faster-whisper applies the VAD chunk offsets before returning).
    # Used to cut exactly-aligned sentence clips for fine-tuning.
    words: list[tuple[str, float, float]] = ()


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
        word_timestamps=True,  # per-word spans → exactly-aligned training clips
    )
    parts: list[str] = []
    words: list[tuple[str, float, float]] = []
    for seg in segments:
        t = (seg.text or "").strip()
        if t:
            parts.append(t)
        for w in seg.words or ():
            wt = (w.word or "").strip()
            if wt:
                words.append((wt, round(float(w.start), 3), round(float(w.end), 3)))
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return Transcription(
        text=text,
        language=info.language,
        language_probability=float(info.language_probability),
        words=words,
    )


# XTTS per-language soft cap is 250 chars for English (check_input_length warns
# above it), and the GPT trainer drops samples past ~200 tokens. Keep every
# training chunk under both with margin: ~240 chars ≈ ~145 tokens.
MAX_SENTENCE_CHARS = 240


@dataclass
class WordChunk:
    """A sentence-like chunk of a transcript, as a span over the word list."""

    text: str
    start_idx: int  # inclusive index into the word list
    end_idx: int  # inclusive index into the word list


def _span_len(words: list[str], s: int, e: int) -> int:
    """Character length of words[s..e] joined with single spaces."""
    return sum(len(w) for w in words[s : e + 1]) + (e - s)


def _break_span(words: list[str], s: int, e: int, max_chars: int) -> list[tuple[int, int]]:
    """Break an over-long span into pieces of at most ``max_chars``.

    Prefers natural clause boundaries (commas / semicolons / dashes), then falls
    back to a hard word window so we never emit a chunk that XTTS would warn on
    or drop.
    """
    if _span_len(words, s, e) <= max_chars:
        return [(s, e)]

    # Try clause boundaries first; only use them if they actually shorten it.
    parts: list[tuple[int, int]] = []
    cs = s
    for i in range(s, e):
        if re.search(r"(?:,|;|—|--)$", words[i]):
            parts.append((cs, i))
            cs = i + 1
    parts.append((cs, e))
    if len(parts) > 1 and max(_span_len(words, a, b) for a, b in parts) < _span_len(words, s, e):
        out: list[tuple[int, int]] = []
        for a, b in parts:
            out.extend(_break_span(words, a, b, max_chars))
        return out

    # Fallback: hard word window.
    out = []
    cs, buf_len = s, 0
    for i in range(s, e + 1):
        wlen = len(words[i])
        cand = wlen if buf_len == 0 else buf_len + 1 + wlen
        if cand > max_chars and buf_len:
            out.append((cs, i - 1))
            cs, buf_len = i, wlen
        else:
            buf_len = cand
    out.append((cs, e))
    return out


def split_words(words_in: list[str], max_chars: int = MAX_SENTENCE_CHARS) -> list[WordChunk]:
    """Split a word list into sentence-like chunks for training pairs.

    Same rules as :func:`split_sentences`, but each chunk carries its span over
    the input words so callers with per-word timestamps can cut exactly-aligned
    audio clips. Splits on terminal punctuation; long run-on chunks (no terminal
    punctuation — common in Whisper output of spoken interviews) are further
    broken on clause boundaries / word windows so no chunk exceeds ``max_chars``.
    Fragments shorter than 3 words are merged with a neighbour when that keeps
    the result under the cap, so the training set stays meaningful.
    """
    words = [w.strip() for w in words_in if w and w.strip()]
    if not words:
        return []

    # 1) Sentence-level split (word spans), then break any over-long sentence further.
    pieces: list[tuple[int, int]] = []
    start = 0
    for i, w in enumerate(words):
        if re.search(r"[.!?…]$", w) or i == len(words) - 1:
            pieces.append((start, i))
            start = i + 1
    broken: list[tuple[int, int]] = []
    for s, e in pieces:
        broken.extend(_break_span(words, s, e, max_chars))

    # 2) Merge tiny trailing fragments into the preceding chunk while under cap.
    chunks: list[tuple[int, int]] = []
    for s, e in broken:
        if (e - s + 1) < 3 and not re.search(r"[.!?…]$", words[e]) and chunks:
            ps, pe = chunks[-1]
            if _span_len(words, ps, e) <= max_chars:
                chunks[-1] = (ps, e)
                continue
        chunks.append((s, e))

    # 3) Drop any remaining slivers (XTTS wants meaningful prompt texts).
    return [
        WordChunk(text=" ".join(words[s : e + 1]), start_idx=s, end_idx=e)
        for s, e in chunks
        if (e - s + 1) >= 3
    ]


def split_sentences(text: str, max_chars: int = MAX_SENTENCE_CHARS) -> list[str]:
    """Split transcript text into sentence-like chunks for training pairs.

    Text-only convenience wrapper over :func:`split_words` (no timing info).
    """
    return [c.text for c in split_words(text.split(), max_chars)]
