"""Compatibility shims for running coqui TTS (Dec 2023) on modern PyTorch.

PyTorch >= 2.6 defaults ``torch.load`` to ``weights_only=True``, which rejects
the config objects stored inside XTTS v2 checkpoints. This context manager
restores the legacy behaviour for its duration.
"""

from __future__ import annotations

import contextlib


@contextlib.contextmanager
def legacy_torch_load():
    import torch

    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


def install_audio_loader() -> None:
    """Replace TTS's ``load_audio`` (torchaudio backend, needs ffmpeg/torchcodec)
    with a PyAV-based loader. Idempotent; call before importing trainer modules."""
    import TTS.tts.models.xtts as xtts_mod

    if getattr(xtts_mod.load_audio, "_vc_patched", False):
        return

    import torch

    from . import audio as A

    def load_audio(audiopath, sampling_rate):
        wav = A.load_audio(str(audiopath), int(sampling_rate))  # np float32 mono @ sr
        tensor = torch.from_numpy(wav).float().unsqueeze(0)     # (1, N)
        if torch.any(tensor > 10) or not torch.any(tensor < 0):
            print(f"Error with {audiopath}. Max={tensor.max()} min={tensor.min()}")
        tensor.clip_(-1, 1)
        return tensor

    load_audio._vc_patched = True  # type: ignore[attr-defined]
    xtts_mod.load_audio = load_audio
