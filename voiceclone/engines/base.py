"""Engine abstraction.

An engine turns (text, reference clip, language) into speech. The default
engine is XTTS v2; the interface is deliberately small so other models
(e.g. instruction-based TTS) can be added later without touching the rest of
the toolkit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SynthesisResult:
    wav: "object"  # np.ndarray float32 mono @ sample_rate
    sample_rate: int
    reference_file: str
    reference_emotion: str
    matched_requested_emotion: bool
    engine: str
    mode: str  # "zero-shot" | "finetuned"
    device: str | None = None  # actual device used ("cuda" | "cpu")
    device_note: str | None = None  # e.g. why the engine fell back to CPU


class Engine(ABC):
    name: str = "base"

    @abstractmethod
    def synthesize(
        self,
        text: str,
        reference_wav_path: str,
        reference_text: str,
        language: str,
        emotion: str = "neutral",
        style: str | None = None,
        finetuned_checkpoint: str | None = None,
    ) -> SynthesisResult:
        """Synthesize ``text`` in the voice of the reference clip.

        ``emotion``/``style`` are hints: engines that support explicit
        sentiment use them directly; others (XTTS v2) realize them through
        reference selection done by the caller.
        """
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional: load model weights ahead of time."""
        return None
