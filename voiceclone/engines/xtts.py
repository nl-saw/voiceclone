"""XTTS v2 engine (zero-shot cloning + fine-tuned checkpoints).

Uses the low-level ``Xtts`` model class directly (same API as the official
fine-tune demo), which lets us:
  * download weights once into our data dir,
  * handle CPML license consent explicitly (no interactive prompts),
  * load either the base weights or a per-voice fine-tuned checkpoint.

Supported languages include English ("en") and Dutch ("nl").
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..config import get_settings
from .base import Engine, SynthesisResult

# Original XTTS v2 weights, fetched directly from HuggingFace.
_BASE_FILES = {
    "dvae": ("https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth", "dvae.pth"),
    "mel": ("https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth", "mel_stats.pth"),
    "vocab": ("https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json", "vocab.json"),
    "model": ("https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth", "model.pth"),
    "config": ("https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json", "config.json"),
    "speakers": ("https://huggingface.co/coqui/XTTS-v2/resolve/main/speakers_xtts.pth", "speakers_xtts.pth"),
}

LICENSE_URL = "https://coqui.ai/page/terms"


class LicenseNotAccepted(Exception):
    def __init__(self) -> None:
        super().__init__(
            "The XTTS v2 weights are distributed under the Coqui Public Model License (non-commercial). "
            f"You must accept it once before first use. See: {LICENSE_URL}\n"
            "Run:  voiceclone init --accept-license"
        )


def base_model_dir() -> Path:
    d = get_settings().models_dir / "XTTS_v2_original_model_files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_weights(allow_download: bool = True) -> dict[str, str]:
    """Ensure the original XTTS v2 files are present; returns local paths."""
    from ..download import fetch

    d = base_model_dir()
    for key, (url, fname) in _BASE_FILES.items():
        target = d / fname
        if not target.is_file() or target.stat().st_size == 0:
            if not allow_download:
                raise RuntimeError(
                    f"Missing XTTS v2 weights under {d}. Run `voiceclone init` to download them."
                )
            print(f"Downloading {fname} ...")
            fetch(url, target)
    return {key: str(d / fname) for key, (_, fname) in _BASE_FILES.items()}


def license_marker() -> Path:
    p = get_settings().models_dir / "LICENSES"
    p.mkdir(parents=True, exist_ok=True)
    return p / "xtts-v2-CPML.accepted"


def accept_license() -> None:
    m = license_marker()
    m.write_text(
        "Accepted the Coqui Public Model License for XTTS v2 (non-commercial use).\n"
        f"{LICENSE_URL}\n"
    )


def is_license_accepted() -> bool:
    return license_marker().exists()


class XttsEngine(Engine):
    name = "xtts-v2"

    def __init__(self) -> None:
        self._model = None
        self._loaded_for: str | None = None  # checkpoint path we loaded for

    # ------------------------------------------------------------------ #
    def _device(self) -> str:
        import torch

        pref = get_settings().device
        if pref == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return pref

    def _load_model(self, finetuned_checkpoint: str | None) -> None:
        """Load base or fine-tuned weights (cached per checkpoint)."""
        key = finetuned_checkpoint or "__base__"
        if self._model is not None and self._loaded_for == key:
            return

        import torch

        torch.set_num_threads(max(1, min(os.cpu_count() or 4, 8)))

        paths = ensure_weights()
        from ..compat import install_audio_loader, legacy_torch_load

        install_audio_loader()  # PyAV instead of torchaudio/ffmpeg backend
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        config = XttsConfig()
        config.load_json(paths["config"])
        ckpt = finetuned_checkpoint if finetuned_checkpoint else paths["model"]
        with legacy_torch_load():  # torch>=2.6 weights_only breaks old checkpoints
            model = Xtts.init_from_config(config)
            model.load_checkpoint(
                config,
                checkpoint_dir=str(base_model_dir()),
                checkpoint_path=ckpt,
                vocab_path=paths["vocab"],
                use_deepspeed=False,
            )
        device = self._device()
        model.to(device)
        model.eval()
        self._model = model
        self._loaded_for = key

    def warmup(self, finetuned_checkpoint: str | None = None) -> None:
        if not is_license_accepted():
            raise LicenseNotAccepted()
        self._load_model(finetuned_checkpoint)

    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        text: str,
        reference_wav_path: str,
        reference_text: str,
        language: str,
        emotion: str = "neutral",
        style: str | None = None,
        finetuned_checkpoint: str | None = None,
        temperature: float | None = None,
        length_penalty: float | None = None,
        repetition_penalty: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        speed: float | None = None,
    ) -> SynthesisResult:
        if not is_license_accepted():
            raise LicenseNotAccepted()

        import torch

        self._load_model(finetuned_checkpoint)
        model = self._model
        config = model.config

        # Generation params: an explicit override always wins. Otherwise we use a
        # toolkit default for temperature (0.5), and the model's own config for
        # everything else.
        temperature = 0.5 if temperature is None else temperature
        length_penalty = config.length_penalty if length_penalty is None else length_penalty
        repetition_penalty = (
            config.repetition_penalty if repetition_penalty is None else repetition_penalty
        )
        top_k = config.top_k if top_k is None else top_k
        top_p = config.top_p if top_p is None else top_p
        speed = 1.0 if speed is None else speed

        with torch.no_grad():
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                audio_path=str(reference_wav_path),
                gpt_cond_len=config.gpt_cond_len,
                max_ref_length=config.max_ref_len,
                sound_norm_refs=config.sound_norm_refs,
            )
            out = model.inference(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                speed=speed,
                enable_text_splitting=True,  # handle long texts safely
            )

        # inference() returns {"wav": np.ndarray (24 kHz), ...}
        wav = np.asarray(out["wav"], dtype=np.float32).reshape(-1)

        return SynthesisResult(
            wav=wav,
            sample_rate=24000,
            reference_file=str(reference_wav_path),
            reference_emotion=emotion,
            matched_requested_emotion=True,
            engine=self.name,
            mode="finetuned" if finetuned_checkpoint else "zero-shot",
        )


_engine: XttsEngine | None = None


def get_xtts_engine() -> XttsEngine:
    global _engine
    if _engine is None:
        _engine = XttsEngine()
    return _engine
