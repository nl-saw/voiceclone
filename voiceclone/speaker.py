"""Speaker-consistency checks on registered samples.

Uses an ECAPA-TDNN speaker embedding (speechbrain, ~80 MB model) to warn when
a voice profile mixes different speakers — the most common way fine-tuning
goes wrong. Everything is lazy and optional: if speechbrain or the model are
unavailable the checks are skipped silently.
"""

from __future__ import annotations

import numpy as np

_ECAPA_SR = 16000

_encoder_cache: dict[str, object] = {}


def _get_encoder():
    if "enc" in _encoder_cache:
        return _encoder_cache["enc"]
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(_get_savedir()),
        run_opts={"device": "cpu"},
    )
    _encoder_cache["enc"] = enc
    return enc


def _get_savedir():
    from .config import get_settings

    p = get_settings().cache_dir / "ecapa"
    p.mkdir(parents=True, exist_ok=True)
    return p


def embed(wav: np.ndarray, sr: int) -> np.ndarray | None:
    """Compute a speaker embedding for a mono float32 waveform (any rate)."""
    try:
        import torch

        from .audio import resample

        enc = _get_encoder()
        audio16 = resample(wav, sr, _ECAPA_SR)
        sig = torch.from_numpy(audio16).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            emb = enc.encode_batch(sig)
        return emb.squeeze().numpy().astype(np.float32)
    except Exception:
        return None


def consistency_report(samples_wavs: list[tuple[str, np.ndarray, int]]) -> list[dict]:
    """Compare speaker embeddings of all samples.

    Returns a list of {id, similarity} (cosine vs the mean embedding), sorted
    by ascending similarity — low values indicate a different speaker.
    Returns [] when embeddings could not be computed.
    """
    items: list[tuple[str, np.ndarray]] = []
    for sid, wav, sr in samples_wavs:
        e = embed(wav, sr)
        if e is not None:
            items.append((sid, e))
    if len(items) < 2:
        return []

    mean = np.mean(np.stack([e for _, e in items]), axis=0)
    norm = lambda x: x / (np.linalg.norm(x) + 1e-9)
    mean = norm(mean)
    out = [
        {"id": sid, "similarity": round(float(np.dot(norm(e), mean)), 3)}
        for sid, e in items
    ]
    out.sort(key=lambda d: d["similarity"])
    return out
