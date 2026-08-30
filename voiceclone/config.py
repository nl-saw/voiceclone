"""Central configuration and path management for the voice cloning toolkit.

Everything lives under a single data directory (default: <project>/data) so the
toolkit is self-contained and easy to back up or move. Override with the
VOICECLONE_DATA environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def project_root() -> Path:
    """The repository / install root (parent of the voiceclone package)."""
    return Path(__file__).resolve().parent.parent


def default_data_dir() -> Path:
    env = os.environ.get("VOICECLONE_DATA")
    if env:
        return Path(env).expanduser().resolve()
    return project_root() / "data"


@dataclass
class Settings:
    """Runtime settings. Loaded from voiceclone/data/settings.json when present."""

    data_dir: Path = field(default_factory=default_data_dir)
    # faster-whisper model size used for auto-transcription of samples.
    # medium is a good accuracy/speed balance for the one-off transcription step;
    # override per-run with `add-sample --whisper-model large-v3` for max word accuracy.
    whisper_model: str = "medium"
    # Languages the user cares about; Whisper is allowed to detect these first.
    languages: list[str] = field(default_factory=lambda: ["en", "nl"])
    # Target sample rate (XTTS v2 native).
    sample_rate: int = 24000
    # Minimum / maximum seconds of a usable reference clip for zero-shot.
    min_ref_seconds: float = 3.0
    max_ref_seconds: float = 15.0
    # Device preference; auto-detects CUDA if available.
    device: str = "auto"

    # --- derived paths -----------------------------------------------------
    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def ensure_dirs(self) -> None:
        for p in (
            self.data_dir,
            self.voices_dir,
            self.models_dir,
            self.output_dir,
            self.cache_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide settings singleton, loading from disk once."""
    global _settings
    if _settings is None:
        s = Settings()
        s.ensure_dirs()
        _settings = s
    return _settings
