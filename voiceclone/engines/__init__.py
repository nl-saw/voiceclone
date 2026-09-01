"""Engine registry and selection.

An *engine* is a concrete TTS model implementing :class:`voiceclone.engines.base.Engine`.
The toolkit ships XTTS v2 out of the box (legacy, unmaintained upstream but kept for
backward compatibility) plus CosyVoice 3 (Apache-2.0, zero-shot + official fine-tune);
each engine sits behind the same small interface, so voices, samples, emotion tags and
fine-tuning pipelines work uniformly. Engines whose dependencies conflict with the
toolkit's own run in a dedicated venv under ``data/engines/<name>/`` and are driven
through a JSON-lines worker process (see :mod:`voiceclone.engines.external`).

Engines are imported lazily: their heavy dependencies only load when the engine is
actually used. If an engine's dependencies are missing, :func:`installed` reports
``False`` and :func:`get_engine` raises a clear error with the install hint.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field


class EngineError(Exception):
    """Unknown engine or unusable engine configuration."""


@dataclass(frozen=True)
class EngineSpec:
    name: str
    module: str  # import path of the engine module
    cls: str  # engine class inside that module
    description: str
    languages: tuple[str, ...]  # ISO codes supported for synthesis
    zero_shot: bool = True
    finetune: bool = False
    probe: tuple[str, ...] = ()  # top-level imports used to detect "installed"
    install_hint: str = ""  # how to install this engine's dependencies
    # Directory template under models_dir holding fine-tune artifacts ("{voice}" is
    # substituted). None when the engine cannot be fine-tuned.
    finetune_root: str | None = None
    # Subdirectory under finetune_root where per-run dirs live ("" = directly).
    runs_subdir: str = ""
    # Glob of checkpoint files inside a run dir (used by storage scanning).
    checkpoint_glob: str = "*.pth"
    # Pre-flight thresholds (GiB) for safe fine-tuning; 0 disables the check.
    min_train_ram_gib: float = 12.0
    min_train_vram_gib: float = 12.0
    # Minimum free VRAM to load this engine for synthesis (0 = no guard).
    min_synth_vram_gib: float = 0.0
    extra: dict = field(default_factory=dict)


REGISTRY: dict[str, EngineSpec] = {
    "xtts-v2": EngineSpec(
        name="xtts-v2",
        module="voiceclone.engines.xtts",
        cls="XttsEngine",
        description="XTTS v2 (Coqui) — legacy default; 17 languages incl. en/nl, zero-shot + fine-tune. Upstream unmaintained.",
        languages=("en", "nl", "fr", "de", "es", "it", "pt", "pl", "tr", "ru", "cs", "ar", "zh-cn", "ja", "hu", "ko", "da"),
        zero_shot=True,
        finetune=True,
        probe=("TTS",),
        install_hint="included in the base install (coqui TTS)",
        finetune_root="{voice}_ft",
        runs_subdir="run/training",
        checkpoint_glob="best_model_*.pth",
        min_train_ram_gib=12.0,
        min_train_vram_gib=12.0,
        min_synth_vram_gib=6.0,
    ),
    "cosyvoice3": EngineSpec(
        name="cosyvoice3",
        module="voiceclone.engines.cosyvoice",
        cls="CosyVoice3Engine",
        description=(
            "FunAudioLLM CosyVoice 3 — Apache-2.0 code+weights, zero-shot + official "
            "fine-tune recipe (9 languages, no Dutch)"
        ),
        languages=("zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru"),
        zero_shot=True,
        finetune=True,
        probe=(),  # external engine: installed() checks the dedicated venv instead
        install_hint=(
            "voiceclone install-engine cosyvoice3 (clones repo + dedicated venv; "
            "~6 GB weights download on first use)"
        ),
        finetune_root="{voice}_ft_cosyvoice3",
        runs_subdir="",
        checkpoint_glob="model_dir/llm.pt",
        min_train_ram_gib=8.0,
        min_train_vram_gib=12.0,
        min_synth_vram_gib=6.0,
        extra={"external": True},
    ),
    "chatterbox": EngineSpec(
        name="chatterbox",
        module="voiceclone.engines.chatterbox",
        cls="ChatterboxEngine",
        description=(
            "Resemble AI Chatterbox Multilingual V3 — MIT code+weights, 23 languages "
            "(incl. Dutch), zero-shot with emotion-intensity control"
        ),
        languages=(
            "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
            "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
        ),
        zero_shot=True,
        finetune=False,  # no official fine-tune recipe yet
        probe=(),  # external engine: installed() checks the dedicated venv instead
        install_hint="voiceclone install-engine chatterbox (dedicated venv; ~3 GB weights on first use)",
        min_train_ram_gib=0.0,
        min_train_vram_gib=0.0,
        min_synth_vram_gib=4.0,
        extra={"external": True},
    ),
}


def engine_names() -> list[str]:
    return sorted(REGISTRY)


def get_spec(name: str) -> EngineSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise EngineError(
            f"Unknown engine '{name}'. Available: {', '.join(engine_names())} "
            f"(see `voiceclone engines`)"
        ) from None


def installed(spec: EngineSpec) -> bool:
    """True when the engine is usable on this machine.

    In-process engines: every probe import succeeds. External engines: their
    dedicated venv exists (created by ``voiceclone install-engine <name>``).
    """
    if spec.extra.get("external"):
        from ..config import get_settings

        return (get_settings().data_dir / "engines" / spec.name / "venv" / "bin" / "python").exists()
    for mod in spec.probe:
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001 — ImportError, ModuleNotFoundError, or broken dep
            return False
    return True


def get_engine(name: str | None = None):
    """Instantiate (lazily) the named engine. ``None`` → configured default."""
    from ..config import get_settings

    name = name or get_settings().default_engine
    spec = get_spec(name)
    if not installed(spec):
        raise EngineError(
            f"Engine '{name}' is not usable on this machine (missing dependencies). "
            f"Install hint: {spec.install_hint or 'see README'}"
        )
    mod = importlib.import_module(spec.module)
    cls = getattr(mod, spec.cls)
    return cls()


def default_engine_name() -> str:
    from ..config import get_settings

    name = get_settings().default_engine
    if name not in REGISTRY:
        raise EngineError(
            f"Configured default engine '{name}' is unknown. "
            f"Available: {', '.join(engine_names())}"
        )
    return name


def list_engines() -> list[dict]:
    """Registry overview for the CLI / web UI."""
    from ..config import get_settings

    default = get_settings().default_engine
    out = []
    for name in engine_names():
        spec = REGISTRY[name]
        out.append({
            "name": name,
            "description": spec.description,
            "languages": list(spec.languages),
            "zero_shot": spec.zero_shot,
            "finetune": spec.finetune,
            "installed": installed(spec),
            "install_hint": spec.install_hint,
            "is_default": name == default,
        })
    return out
