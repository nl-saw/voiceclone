"""Central configuration and path management for the voice cloning toolkit.

Everything lives under a single data directory (default: <project>/data) so the
toolkit is self-contained and easy to back up or move. Override with the
VOICECLONE_DATA environment variable.
"""

from __future__ import annotations

import json
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
    """Runtime settings. Persisted to <data_dir>/settings.json; a missing file is
    created with these defaults (default_engine = "xtts-v2") on first use."""

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
    # Default TTS engine (see voiceclone/engines registry). Override per-run with
    # `--engine` on synthesize/train.
    default_engine: str = "xtts-v2"

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


# Known settings.json keys → attribute types (unknown keys are ignored).
_KNOWN_KEYS: dict[str, type] = {
    "whisper_model": str,
    "languages": list,
    "sample_rate": int,
    "min_ref_seconds": float,
    "max_ref_seconds": float,
    "device": str,
    "default_engine": str,
}

_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide settings singleton, loading from disk once.

    Values in ``<data_dir>/settings.json`` override the defaults (e.g.
    ``{"default_engine": "cosyvoice3"}``). A missing file is created with the
    defaults on first use; a broken one is ignored (never overwritten).
    """
    global _settings
    if _settings is None:
        s = Settings()
        cfg_file = s.data_dir / "settings.json"
        if cfg_file.is_file():
            try:
                data = json.loads(cfg_file.read_text())
                for key, typ in _KNOWN_KEYS.items():
                    if key in data and isinstance(data[key], typ):
                        setattr(s, key, data[key])
            except (json.JSONDecodeError, OSError):
                pass  # broken settings file must not break the toolkit
        s.ensure_dirs()
        if not cfg_file.is_file():
            try:
                payload = {key: getattr(s, key) for key in _KNOWN_KEYS}
                cfg_file.write_text(json.dumps(payload, indent=2) + "\n")
            except OSError:
                pass  # an unwritable data dir must not break the toolkit
        _settings = s
    return _settings
