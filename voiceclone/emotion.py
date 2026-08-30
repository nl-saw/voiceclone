"""Sentiment / emotion control for synthesis.

XTTS v2 carries prosody (and to a large degree emotional tone) from the
reference clip, so sentiment control is implemented by *reference selection*:

  * Every registered sample can carry an emotion tag (default: neutral).
  * When synthesizing with ``--emotion sad`` we pick the voice's most-sad
    tagged sample as the reference clip.
  * Free-text style ("whisper this, very calm") is mapped onto the closest
    preset emotion via keyword matching.

This is a practical, honest mechanism: it works out of the box with any set
of samples (neutral fallback) and improves as the user tags clips with
emotion.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

PRESET_EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "calm",
    "excited",
    "fearful",
    "surprised",
]

# Keyword → emotion mapping for free-text style descriptions.
# Order matters: first match wins, so more specific words come first.
_STYLE_KEYWORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(excited|excitement|thrilled|pumped|elated|hyped)\b", re.I), "excited"),
    (re.compile(r"\b(angry|anger|furious|irritated|annoyed|shouting|yelling|mad)\b", re.I), "angry"),
    (re.compile(r"\b(sad|sorrow|sadness|grief|gloomy|melanchol|downbeat|depressed|mournful)\b", re.I), "sad"),
    (re.compile(r"\b(happy|cheerful|joyful|joy|upbeat|bright|jolly)\b", re.I), "happy"),
    (re.compile(r"\b(fear|fearful|scared|afraid|anxious|nervous|tense|terrified)\b", re.I), "fearful"),
    (re.compile(r"\b(surpris|astonish|amazed|shock(ed)?|incredul)\b", re.I), "surprised"),
    (re.compile(r"\b(calm|relaxed|soothing|gentle|mellow|soft|quiet|whisper|laid.back|sleepy)\b", re.I), "calm"),
]


def map_style_to_emotion(style: str | None) -> str | None:
    """Map a free-text style description onto the closest preset emotion.

    Returns None when nothing matches (caller then falls back to neutral).
    """
    if not style:
        return None
    for pattern, emotion in _STYLE_KEYWORDS:
        if pattern.search(style):
            return emotion
    return None


@dataclass
class ReferenceChoice:
    sample_id: str
    file: str
    transcript: str
    emotion: str
    duration_s: float
    matched_requested_emotion: bool


def select_reference(
    samples: list[dict],
    emotion: str,
    min_seconds: float = 3.0,
    max_seconds: float = 15.0,
) -> ReferenceChoice | None:
    """Pick the best reference sample for a requested emotion.

    Scoring (higher is better):
      1. exact emotion tag match
      2. duration in [min, max] (XTTS sweet spot ~6-10 s)
      3. longer is slightly better (more voice information), capped
      4. stable deterministic tie-break on sample id

    Falls back to neutral samples when no tagged sample exists.
    """
    emotion = (emotion or "neutral").lower()
    if emotion not in PRESET_EMOTIONS:
        emotion = "neutral"

    def score(s: dict) -> tuple[int, float, float, str]:
        tag = (s.get("emotion") or "neutral").lower()
        matched = 1 if tag == emotion else 0
        d = float(s.get("duration_s") or 0.0)
        in_range = 1.0 if min_seconds <= d <= max_seconds else 0.0
        # mild preference for mid-range durations
        dur_score = min(d, max_seconds) / max_seconds
        return (matched, in_range, dur_score, s.get("id", ""))

    candidates = [s for s in samples if (s.get("transcript") or "").strip()]
    if not candidates:
        return None

    best = max(candidates, key=score)
    tag = (best.get("emotion") or "neutral").lower()
    return ReferenceChoice(
        sample_id=best["id"],
        file=best["file"],
        transcript=best["transcript"],
        emotion=tag,
        duration_s=float(best.get("duration_s") or 0.0),
        matched_requested_emotion=tag == emotion,
    )


def deterministic_ref_index(text: str, voice_name: str, n_samples: int) -> int:
    """Stable pseudo-random index (for rotating references when the user
    wants variety)."""
    h = hashlib.sha256(f"{voice_name}|{text}".encode()).digest()
    return int.from_bytes(h[:4], "big") % max(1, n_samples)
