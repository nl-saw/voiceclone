"""Voice profile storage.

A *voice* is a directory under ``data/voices/<name>/`` containing:

  voice.json    metadata + sample list + emotion tags + finetune info
  samples/      normalized 24 kHz mono WAV clips (the user's raw material)
  train/        prepared fine-tuning dataset (wavs/ + CSVs), when built

Everything is plain JSON + WAV so voices are portable and inspectable.
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import audio as A
from .config import get_settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s or "voice"


class VoiceError(Exception):
    pass


@dataclass
class Sample:
    id: str
    file: str  # relative to voice dir, e.g. samples/s001.wav
    transcript: str
    language: str
    emotion: str = "neutral"
    duration_s: float = 0.0
    added_at: str = field(default_factory=_now)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "transcript": self.transcript,
            "language": self.language,
            "emotion": self.emotion,
            "duration_s": self.duration_s,
            "added_at": self.added_at,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(**{k: d.get(k, v) for k, v in {
            "id": "", "file": "", "transcript": "", "language": "en",
            "emotion": "neutral", "duration_s": 0.0, "added_at": _now(), "note": "",
        }.items() if k in d})


@dataclass
class Voice:
    name: str
    dir: Path
    created_at: str = field(default_factory=_now)
    samples: list[Sample] = field(default_factory=list)
    finetuned: dict | None = None  # {"checkpoint": ..., "trained_at": ..., "epochs": ...}

    @property
    def total_seconds(self) -> float:
        return sum(s.duration_s for s in self.samples)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "samples": [s.to_dict() for s in self.samples],
            "finetuned": self.finetuned,
        }


# --------------------------------------------------------------------------- #
# Store operations
# --------------------------------------------------------------------------- #

def voice_dir(name: str) -> Path:
    return get_settings().voices_dir / slugify(name)


def create_voice(name: str, overwrite: bool = False) -> Voice:
    d = voice_dir(name)
    if d.exists() and not overwrite:
        raise VoiceError(f"Voice '{name}' already exists ({d}). Use --overwrite to replace.")
    if d.exists():
        shutil.rmtree(d)
    (d / "samples").mkdir(parents=True, exist_ok=True)
    v = Voice(name=slugify(name), dir=d)
    _save(v)
    return v


def load_voice(name: str) -> Voice:
    d = voice_dir(name)
    meta = d / "voice.json"
    if not meta.exists():
        raise VoiceError(f"Voice '{name}' not found. Run `voiceclone add-sample {name} <file>` first.")
    data = json.loads(meta.read_text())
    v = Voice(
        name=data["name"],
        dir=d,
        created_at=data.get("created_at", _now()),
        samples=[Sample.from_dict(s) for s in data.get("samples", [])],
        finetuned=data.get("finetuned"),
    )
    return v


def list_voices() -> list[Voice]:
    root = get_settings().voices_dir
    out: list[Voice] = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        meta = d / "voice.json"
        if d.is_dir() and meta.exists():
            try:
                data = json.loads(meta.read_text())
                out.append(Voice(
                    name=data["name"], dir=d,
                    created_at=data.get("created_at", _now()),
                    samples=[Sample.from_dict(s) for s in data.get("samples", [])],
                    finetuned=data.get("finetuned"),
                ))
            except Exception:
                continue
    return out


def _save(v: Voice) -> None:
    (v.dir / "voice.json").write_text(json.dumps(v.to_dict(), indent=2, ensure_ascii=False))


def next_sample_id(v: Voice) -> str:
    nums = [int(re.sub(r"\D", "", s.id) or 0) for s in v.samples]
    return f"s{(max(nums) if nums else 0) + 1:03d}"


def add_samples(
    name: str,
    files: list[str],
    language: str | None = None,
    emotion: str = "neutral",
    note: str = "",
    whisper_model: str | None = None,
    transcriber=None,
) -> tuple[Voice, list[dict]]:
    """Register audio files as samples of a voice.

    Each file is decoded to 24 kHz mono, trimmed, normalized, stored under
    ``samples/``, and transcribed (``transcriber`` is a callable
    ``wav, sr -> (text, lang)``; when None the default faster-whisper is used).

    Returns (voice, per-file reports).
    """
    from .emotion import PRESET_EMOTIONS

    if emotion not in PRESET_EMOTIONS:
        raise VoiceError(f"Unknown emotion '{emotion}'. Choose from: {', '.join(PRESET_EMOTIONS)}")

    if language in ("auto", "", None):
        language = None  # Whisper auto-detects

    v = load_voice(name) if voice_dir(name).exists() else create_voice(name)
    reports: list[dict] = []

    for f in files:
        p = Path(f).expanduser()
        if not p.exists():
            reports.append({"file": str(f), "ok": False, "error": "file not found"})
            continue
        if not A.is_supported_audio(str(p)):
            reports.append({"file": str(f), "ok": False, "error": f"unsupported format (want {', '.join(A.SUPPORTED_EXTS)})"})
            continue

        sid = next_sample_id(v)
        out_rel = f"samples/{sid}.wav"
        try:
            wav = A.load_audio(str(p), get_settings().sample_rate)
            wav = A.trim_silence(wav, get_settings().sample_rate)
            wav = A.normalize_peak(wav)
            dur = A.duration_seconds(wav, get_settings().sample_rate)

            if dur < 1.0:
                reports.append({"file": str(f), "ok": False, "error": f"too short after trimming ({dur:.1f}s)"})
                continue

            # Transcribe
            text, lang = "", language or ""
            if transcriber is not None:
                text, lang = transcriber(wav, get_settings().sample_rate)
            else:
                from .transcribe import transcribe_wav

                model_size = whisper_model or get_settings().whisper_model
                t = transcribe_wav(wav, get_settings().sample_rate, language=language, model_size=model_size)
                text, lang = t.text, t.language
            if not lang:
                lang = "en"

            A.save_wav(str(v.dir / out_rel), wav, get_settings().sample_rate)

            v.samples.append(Sample(
                id=sid, file=out_rel, transcript=text, language=lang,
                emotion=emotion, duration_s=round(dur, 2), note=note,
            ))
            reports.append({
                "file": str(f), "ok": True, "sample_id": sid,
                "duration_s": round(dur, 2), "language": lang,
                "transcript": text[:160],
            })
        except Exception as e:  # noqa: BLE001 — one bad file shouldn't kill the batch
            (v.dir / out_rel).unlink(missing_ok=True)
            reports.append({"file": str(f), "ok": False, "error": f"{type(e).__name__}: {e}"})

    _save(v)
    return v, reports


def tag_sample(name: str, sample_id: str, emotion: str | None = None, note: str | None = None) -> Voice:
    from .emotion import PRESET_EMOTIONS

    v = load_voice(name)
    s = next((x for x in v.samples if x.id == sample_id), None)
    if s is None:
        raise VoiceError(f"Sample '{sample_id}' not found in voice '{name}'.")
    if emotion is not None:
        if emotion not in PRESET_EMOTIONS:
            raise VoiceError(f"Unknown emotion '{emotion}'. Choose from: {', '.join(PRESET_EMOTIONS)}")
        s.emotion = emotion
    if note is not None:
        s.note = note
    _save(v)
    return v


def remove_sample(name: str, sample_id: str) -> Voice:
    v = load_voice(name)
    before = len(v.samples)
    s = next((x for x in v.samples if x.id == sample_id), None)
    if s is None:
        raise VoiceError(f"Sample '{sample_id}' not found in voice '{name}'.")
    try:
        (v.dir / s.file).unlink(missing_ok=True)
    except OSError:
        pass
    v.samples = [x for x in v.samples if x.id != sample_id]
    assert len(v.samples) == before - 1
    _save(v)
    return v


def set_finetuned(name: str, info: dict | None) -> Voice:
    v = load_voice(name)
    v.finetuned = info
    _save(v)
    return v
