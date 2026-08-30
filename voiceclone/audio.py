"""Audio I/O helpers.

Decode arbitrary input formats (mp3, wav, flac, m4a, ogg, opus, webm) to
24 kHz mono float32 PCM using PyAV — no system ffmpeg required.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

SUPPORTED_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".webm")


def is_supported_audio(path: str) -> bool:
    return path.lower().endswith(SUPPORTED_EXTS)


def load_audio(path: str, target_sr: int = 24000) -> np.ndarray:
    """Decode any supported audio file to mono float32 at ``target_sr``.

    Returns a 1-D numpy array in [-1, 1].

    Corrupt streams (bad packets from interrupted downloads etc.) are
    salvaged: everything that decodes cleanly before the first bad packet is
    kept, with a warning. If nothing decodes, the original error is raised.
    """
    import warnings

    import av

    container = av.open(str(path))
    chunks: list[np.ndarray] = []
    try:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
        try:
            for frame in container.decode(stream):
                for rf in resampler.resample(frame):
                    arr = rf.to_ndarray()  # shape (1, samples)
                    chunks.append(arr.reshape(-1))
        except Exception as e:  # noqa: BLE001 — salvage partial decode of corrupt files
            if not chunks:
                raise
            kept_s = sum(len(c) for c in chunks) / target_sr
            warnings.warn(
                f"{path}: audio stream is corrupted ({type(e).__name__}); "
                f"keeping the {kept_s:.0f}s that decoded cleanly before the damage.",
                stacklevel=2,
            )
    finally:
        container.close()

    if not chunks:
        raise ValueError(f"No audio data found in {path}")
    return np.concatenate(chunks).astype(np.float32)


def save_wav(path: str, wav: np.ndarray, sr: int = 24000) -> None:
    """Save float32 mono PCM as a 16-bit PCM WAV file."""
    sf.write(str(path), wav.astype(np.float32), sr, subtype="PCM_16")


def resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav
    import av

    frame = av.AudioFrame.from_ndarray(
        wav.reshape(1, -1).astype(np.float32), format="flt", layout="mono"
    )
    frame.sample_rate = src_sr
    resampler = av.AudioResampler(format="flt", layout="mono", rate=dst_sr)
    out = [rf.to_ndarray().reshape(-1) for rf in resampler.resample(frame)]
    tail = list(resampler.resample(None))
    out.extend(rf.to_ndarray().reshape(-1) for rf in tail)
    return np.concatenate(out).astype(np.float32) if out else wav


def duration_seconds(wav: np.ndarray, sr: int) -> float:
    return len(wav) / float(sr)


def rms_db(wav: np.ndarray) -> float:
    """Root-mean-square level in dBFS (approximate)."""
    if wav.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))
    if rms <= 1e-9:
        return -120.0
    return float(20.0 * np.log10(rms))


def trim_silence(
    wav: np.ndarray,
    sr: int,
    pad_ms: int = 150,
    threshold_db: float = -45.0,
) -> np.ndarray:
    """Cut leading/trailing near-silence using a simple RMS gate.

    Keeps at least ``pad_ms`` of audio on each side of the voiced region.
    Falls back to the original signal if nothing passes the threshold.
    """
    if wav.size == 0:
        return wav
    frame = max(1, int(sr * 0.02))  # 20 ms frames
    n_frames = max(1, len(wav) // frame)
    rms = np.array(
        [np.sqrt(np.mean(np.square(wav[i * frame : (i + 1) * frame]))) for i in range(n_frames)]
    )
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 1e-9:
        return wav
    thresh = peak * (10 ** (threshold_db / 20.0))
    voiced = np.where(rms >= thresh)[0]
    if voiced.size == 0:
        return wav
    start = max(0, int(voiced[0] * frame - pad_ms / 1000.0 * sr))
    end = min(len(wav), int(voiced[-1] * frame + frame + pad_ms / 1000.0 * sr))
    return wav[start:end]


def normalize_peak(wav: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    """Scale the signal so its peak sits at ``target_dbfs`` (default -3 dB)."""
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak <= 1e-9:
        return wav
    target = 10 ** (target_dbfs / 20.0)
    return (wav * (target / peak)).astype(np.float32)
