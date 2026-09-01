"""Command-line interface for the voice cloning toolkit.

Commands:
  init            download models + accept licenses (one-time setup)
  add-sample      register audio files as samples of a voice
  voices          list registered voices
  voice           show one voice in detail
  tag             set an emotion tag on a sample
  synthesize      generate speech with a cloned voice (+ sentiment)
  train           fine-tune a per-voice model (engine-specific)
  engines         list available TTS engines + install status
  install-engine  install an external engine (repo + dedicated venv + deps)
  serve           start the web UI
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import get_settings
from .emotion import PRESET_EMOTIONS, map_style_to_emotion
from .voices import VoiceError, list_voices, load_voice

console = Console()


# --------------------------------------------------------------------------- #
# init
# --------------------------------------------------------------------------- #

def cmd_init(args: argparse.Namespace) -> int:
    settings = get_settings()
    settings.ensure_dirs()
    console.print(f"[bold]Data dir:[/bold] {settings.data_dir}")

    if args.accept_license:
        from .engines.xtts import LICENSE_URL, accept_license

        accept_license()
        console.print("[green]✔[/green] XTTS v2 (CPML, non-commercial) license accepted. "
                      f"Terms: {LICENSE_URL}")
    else:
        from .engines.xtts import is_license_accepted

        if not is_license_accepted():
            console.print(
                "[yellow]XTTS v2 license not accepted yet.[/yellow] Run:\n"
                "  voiceclone init --accept-license\n"
                "(weights are non-commercial licensed; see the README)"
            )

    # Pre-download XTTS v2 weights + whisper model so first synthesis is fast.
    if args.download:
        console.print("[bold]Downloading XTTS v2 weights (~1.9 GB) ...[/bold]")
        from .engines.xtts import ensure_weights, is_license_accepted

        if not is_license_accepted():
            raise SystemExit("Refusing to download XTTS v2 weights without license acceptance. "
                             "Run: voiceclone init --accept-license")
        t0 = time.time()
        ensure_weights(allow_download=True)
        console.print(f"[green]✔[/green] XTTS v2 weights ready ({time.time() - t0:.0f}s)")

        console.print(f"[bold]Downloading Whisper '{settings.whisper_model}' (transcription) ...[/bold]")
        from .transcribe import get_whisper

        t0 = time.time()
        get_whisper(settings.whisper_model)
        console.print(f"[green]✔[/green] Whisper ready ({time.time() - t0:.0f}s)")

        # pre-download weights of installed external engines (Chatterbox pulls
        # its ~3 GB automatically on first synthesis; CosyVoice is explicit)
        from .engines import get_spec, installed as _installed

        cv3 = get_spec("cosyvoice3")
        if _installed(cv3):
            console.print("[bold]Downloading CosyVoice 3 weights (~6 GB) ...[/bold]")
            from .engines.cosyvoice import ensure_weights

            t0 = time.time()
            ensure_weights(lambda m: console.print(f"  {m}"))
            console.print(f"[green]✔[/green] CosyVoice 3 weights ready ({time.time() - t0:.0f}s)")

    console.print("[green]Setup complete.[/green]")
    return 0


# --------------------------------------------------------------------------- #
# voices / samples
# --------------------------------------------------------------------------- #

def cmd_add_sample(args: argparse.Namespace) -> int:
    from .voices import add_samples

    files = args.files
    if not files:
        console.print("[red]No input files given.[/red]")
        return 2

    try:
        voice, reports = add_samples(
            args.voice,
            files,
            language=args.lang,
            emotion=args.emotion,
            note=args.note or "",
            whisper_model=args.whisper_model,
        )
    except VoiceError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    for r in reports:
        if r["ok"]:
            console.print(
                f"[green]✔[/green] {r['file']} → {r['sample_id']} "
                f"({r['duration_s']:.1f}s, {r['language']}, emotion={args.emotion})\n"
                f"    transcript: {r['transcript']}"
            )
        else:
            console.print(f"[red]✘[/red] {r['file']}: {r['error']}")

    total = voice.total_seconds
    n = len(voice.samples)
    console.print(f"\nVoice '{voice.name}': {n} samples, {total:.0f}s total audio.")
    if total < 60:
        console.print("[yellow]Tip:[/yellow] fine-tuning quality improves a lot with ≥ 1-5 min of clean speech.")
    return 0


def cmd_voices(args: argparse.Namespace) -> int:
    voices = list_voices()
    if not voices:
        console.print("No voices yet. Register one:\n"
                      "  voiceclone add-sample alice sample1.wav sample2.mp3")
        return 0
    table = Table(title="Registered voices")
    table.add_column("name", style="bold")
    table.add_column("samples")
    table.add_column("audio (s)")
    table.add_column("languages")
    table.add_column("emotions tagged")
    table.add_column("finetuned")
    for v in voices:
        langs = sorted({s.language for s in v.samples})
        emotions = sorted({s.emotion for s in v.samples if s.emotion != "neutral"}) or ["-"]
        table.add_row(
            v.name,
            str(len(v.samples)),
            f"{v.total_seconds:.0f}",
            ", ".join(langs),
            ", ".join(emotions),
            "✔" if v.finetuned else "",
        )
    console.print(table)
    return 0


def cmd_voice(args: argparse.Namespace) -> int:
    try:
        v = load_voice(args.voice)
    except VoiceError as e:
        console.print(f"[red]{e}[/red]")
        return 1
    console.print(f"[bold]{v.name}[/bold] — {len(v.samples)} samples, {v.total_seconds:.0f}s total")
    if v.finetuned:
        console.print(f"finetuned checkpoint: {v.finetuned.get('checkpoint')}")
    table = Table()
    for col in ("id", "emotion", "lang", "dur(s)", "transcript"):
        table.add_column(col)
    for s in v.samples:
        table.add_row(s.id, s.emotion, s.language, f"{s.duration_s:.1f}", (s.transcript or "")[:70])
    console.print(table)
    return 0


def cmd_tag(args: argparse.Namespace) -> int:
    from .voices import tag_sample

    try:
        v = tag_sample(args.voice, args.sample_id, emotion=args.emotion, note=args.note)
    except VoiceError as e:
        console.print(f"[red]{e}[/red]")
        return 1
    s = next(x for x in v.samples if x.id == args.sample_id)
    console.print(f"[green]✔[/green] {s.id} → emotion={s.emotion}")
    return 0


def cmd_remove_sample(args: argparse.Namespace) -> int:
    from .voices import remove_sample

    try:
        v = remove_sample(args.voice, args.sample_id)
    except VoiceError as e:
        console.print(f"[red]{e}[/red]")
        return 1
    console.print(f"[green]✔[/green] removed {args.sample_id}; {len(v.samples)} samples left.")
    return 0


# --------------------------------------------------------------------------- #
# synthesize
# --------------------------------------------------------------------------- #

def cmd_synthesize(args: argparse.Namespace) -> int:
    from .engines import EngineError, get_spec
    from .synthesize import synthesize

    try:
        voice = load_voice(args.voice)
    except VoiceError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    try:
        spec = get_spec(args.engine)
    except EngineError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    text = args.text
    if not text and sys.stdin.isatty():
        console.print("[red]No text given (and stdin is a TTY). Pass text as an argument or pipe it.[/red]")
        return 2
    if not text:
        text = sys.stdin.read()

    emotion = args.emotion
    if args.style and (emotion is None or emotion == "neutral"):
        mapped = map_style_to_emotion(args.style)
        if mapped:
            emotion = mapped

    out_path = Path(args.output) if args.output else None

    console.print(f"Synthesizing with voice [bold]{voice.name}[/bold] "
                  f"(engine={spec.name}, emotion={emotion or 'auto'}, mode={args.mode}) ...")
    t0 = time.time()
    try:
        outcome = synthesize(
            voice=voice,
            text=text,
            emotion=emotion,
            style=args.style,
            language=args.lang,
            engine_mode=args.mode,
            engine_name=spec.name,
            output_path=out_path,
            temperature=args.temp,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k,
            top_p=args.top_p,
            speed=args.speed,
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Synthesis failed:[/red] {e}")
        return 1

    dt = time.time() - t0
    res = outcome.result
    dur = len(res.wav) / res.sample_rate
    console.print(
        f"[green]✔[/green] {dur:.1f}s of speech in {dt:.1f}s "
        f"({dur / max(dt, 1e-6):.1f}x real-time) — mode={res.mode}, engine={res.engine}\n"
        f"    reference: {Path(res.reference_file).name} (tagged '{outcome.resolved_emotion}', "
        f"requested '{outcome.requested_emotion}'{'' if outcome.resolved_emotion == outcome.requested_emotion else ' — fell back to closest available'})\n"
        f"    output: [bold]{outcome.output_path}[/bold]"
    )
    return 0


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #

def cmd_train(args: argparse.Namespace) -> int:
    from .engines import EngineError, get_spec
    from .train import record_finetune, run_finetune
    from .voices import load_voice

    try:
        spec = get_spec(args.engine)
    except EngineError as e:
        console.print(f"[red]{e}[/red]")
        return 2
    if not spec.finetune:
        console.print(f"[red]Engine '{spec.name}' does not support fine-tuning.[/red]")
        return 2

    try:
        v = load_voice(args.voice)
    except VoiceError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    if not v.samples:
        console.print("[red]Voice has no samples. Add some first.[/red]")
        return 1
    total = v.total_seconds
    console.print(
        f"Fine-tuning voice [bold]{v.name}[/bold]: {len(v.samples)} samples, {total:.0f}s audio.\n"
        + ("" if args.dry_run else
           "[yellow]On CPU this can take hours. On a GPU machine it takes minutes-to-an-hour.[/yellow]\n")
    )

    settings = get_settings()
    log_path = settings.data_dir / "logs" / f"train_{v.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        report = run_finetune(
            v.name,
            engine=spec.name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            dry_run=args.dry_run,
            force=args.force,
            precision=args.precision,
            lr=args.lr,
            log_path=log_path,
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Training failed:[/red] {e}\nSee log: {log_path}")
        return 1

    if not args.dry_run and report.checkpoint:
        record_finetune(v.name, report, engine=spec.name)
        console.print(
            f"[green]✔[/green] Fine-tuned ({spec.name}). Checkpoint: {report.checkpoint}\n"
            f"Synthesize with it:  voiceclone synthesize --voice {v.name} --engine {spec.name} --mode finetuned \"text\""
        )
    else:
        console.print(f"[green]✔[/green] Dataset prepared under {report.output_dir}")
    return 0


# --------------------------------------------------------------------------- #
# engines
# --------------------------------------------------------------------------- #

def cmd_engines(args: argparse.Namespace) -> int:
    from .engines import list_engines

    table = Table(title="TTS engines")
    table.add_column("engine", style="bold")
    table.add_column("installed")
    table.add_column("zero-shot")
    table.add_column("fine-tune")
    table.add_column("languages")
    table.add_column("notes")
    for e in list_engines():
        table.add_row(
            e["name"] + (" (default)" if e["is_default"] else ""),
            "[green]✔[/green]" if e["installed"] else "[red]✘ not installed[/red]",
            "✔" if e["zero_shot"] else "",
            "✔" if e["finetune"] else "",
            ", ".join(e["languages"]),
            (e["install_hint"] or e["description"])[:80],
        )
    console.print(table)
    return 0


# --------------------------------------------------------------------------- #
# install-engine
# --------------------------------------------------------------------------- #

def cmd_install_engine(args: argparse.Namespace) -> int:
    import importlib

    from .engines import EngineError, get_spec

    try:
        spec = get_spec(args.name)
    except EngineError as e:
        console.print(f"[red]{e}[/red]")
        return 2
    if not spec.extra.get("external"):
        console.print(f"[yellow]Engine '{spec.name}' is part of the base install — nothing to do.[/yellow]")
        return 0

    mod = importlib.import_module(spec.module)

    def logline(msg: str) -> None:
        console.print(f"  {msg}")

    console.print(f"[bold]Installing engine '{spec.name}'[/bold] (clones the repo, creates a "
                  f"dedicated venv, installs dependencies — this can take a while and use several GB)\n")
    t0 = time.time()
    try:
        mod.ensure_installed(logline=logline)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Install failed:[/red] {e}")
        return 1
    console.print(f"[green]✔[/green] Engine '{spec.name}' installed in {time.time() - t0:.0f}s")
    return 0


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #

def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    app = create_app()
    console.print(f"[bold]Web UI:[/bold] http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voiceclone", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init", help="one-time setup (download models, accept license)")
    sp.add_argument("--accept-license", action="store_true", help="accept the XTTS v2 (CPML, non-commercial) license")
    sp.add_argument("--download", action="store_true", help="pre-download XTTS v2 + Whisper models now")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("add-sample", help="register audio files as samples of a voice")
    sp.add_argument("voice", help="voice name (e.g. alice)")
    sp.add_argument("files", nargs="+", help=".wav/.mp3/.flac/.m4a files with the target speaker's voice")
    sp.add_argument("--lang", default="auto", help="force language ISO code (default: auto-detect by Whisper)")
    sp.add_argument("--emotion", default="neutral", choices=PRESET_EMOTIONS, help="emotion tag for these samples")
    sp.add_argument("--note", default="", help="optional note stored with the samples")
    sp.add_argument(
        "--whisper-model",
        default=None,
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="faster-whisper transcription model (default: config 'medium'); bigger = more accurate words but a slower one-off run",
    )
    sp.set_defaults(func=cmd_add_sample)

    sp = sub.add_parser("voices", help="list registered voices")
    sp.set_defaults(func=cmd_voices)

    sp = sub.add_parser("voice", help="show one voice in detail")
    sp.add_argument("voice")
    sp.set_defaults(func=cmd_voice)

    sp = sub.add_parser("tag", help="set an emotion tag on a sample")
    sp.add_argument("voice")
    sp.add_argument("sample_id", help="e.g. s001")
    sp.add_argument("--emotion", required=True, choices=PRESET_EMOTIONS)
    sp.add_argument("--note", default=None)
    sp.set_defaults(func=cmd_tag)

    sp = sub.add_parser("remove-sample", help="delete a sample from a voice")
    sp.add_argument("voice")
    sp.add_argument("sample_id")
    sp.set_defaults(func=cmd_remove_sample)

    sp = sub.add_parser("synthesize", help="generate speech with a cloned voice")
    sp.add_argument("text", nargs="?", help="text to speak (or pipe via stdin)")
    sp.add_argument("--voice", required=True)
    sp.add_argument("--emotion", choices=PRESET_EMOTIONS, default=None, help="sentiment preset")
    sp.add_argument("--style", default=None, help="free-text style ('whisper this', 'angry and loud') — mapped to the closest preset")
    sp.add_argument("--lang", default="auto", help="force language ISO code (default: auto-detect from samples; engine validates)")
    sp.add_argument("--mode", choices=["auto", "zero-shot", "finetuned"], default="auto")
    sp.add_argument("--engine", default=None, help="TTS engine to use (default: configured default; see `voiceclone engines`)")
    sp.add_argument("-o", "--output", default=None, help="output .wav path (default: data/output/...)")
    # Generation tuning — all optional; when omitted the XTTS model's own defaults are used.
    sp.add_argument("--temp", type=float, default=None, help="sampling temperature (default 0.5; lower = more committed, fewer skipped words)")
    sp.add_argument("--length-penalty", type=float, default=None, help="length penalty (default 1.0; raise to ~1.2-1.5 to reduce dropped words)")
    sp.add_argument("--repetition-penalty", type=float, default=None, help="repetition penalty (default 5.0; higher = less stuttering/repeats)")
    sp.add_argument("--top-k", type=int, default=None, help="top-k sampling (default 50; lower = more focused)")
    sp.add_argument("--top-p", type=float, default=None, help="nucleus top-p (default 0.85; lower = more focused)")
    sp.add_argument("--speed", type=float, default=None, help="speech speed (default 1.0; ~0.9 = clearer/slower, >1 = faster but riskier drops)")
    sp.set_defaults(func=cmd_synthesize)

    sp = sub.add_parser("engines", help="list available TTS engines + install status")
    sp.set_defaults(func=cmd_engines)

    sp = sub.add_parser("install-engine", help="install an external engine (repo + dedicated venv + deps)")
    from .engines import engine_names as _engine_names

    sp.add_argument("name", choices=_engine_names())
    sp.set_defaults(func=cmd_install_engine)

    sp = sub.add_parser("train", help="fine-tune a per-voice model (engine-specific)")
    sp.add_argument("voice")
    sp.add_argument("--engine", default=None, help="TTS engine to fine-tune (default: configured default; see `voiceclone engines`)")
    sp.add_argument("--epochs", type=int, default=5)
    sp.add_argument("--batch-size", type=int, default=1)
    sp.add_argument("--grad-accum", type=int, default=4, help="gradient accumulation steps (effective batch = bs * accum)")
    sp.add_argument("--lr", type=float, default=None, help="learning rate (default 4e-06; lower = closer to base voice quality, higher = more voice similarity)")
    sp.add_argument("--dry-run", action="store_true", help="only prepare the training dataset")
    sp.add_argument("--force", action="store_true", help="start even if free RAM looks too low (OOM risk)")
    sp.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp32"],
        default="auto",
        help="training precision: auto = bfloat16 when a CUDA GPU is present, else float32 (default)",
    )
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("serve", help="start the web UI")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8321)
    sp.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
