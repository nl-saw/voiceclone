# voiceclone 🎙️

A local **voice cloning toolkit**: register a person's voice from plain audio
files (`.wav`/`.mp3`/…), optionally fine-tune a dedicated model per voice, and
synthesize speech **with sentiment control** (happy / sad / angry / calm / …),
fully offline after setup.

Multi-engine: ships **XTTS v2** (legacy default), **CosyVoice 3** (Apache-2.0,
zero-shot + official fine-tune recipe) and **Chatterbox Multilingual V3** (MIT,
23 languages incl. Dutch). Pick per synthesis or per training run with
`--engine`; the default is configurable (`data/settings.json`).

```
┌─────────────┐   samples    ┌──────────────────┐   text + emotion   ┌────────────┐
│ .wav/.mp3   │─────────────▶│ voice profile    │───────────────────▶│  speech    │
│ (any length)│              │ + transcripts    │   TTS engine       │  .wav out  │
└─────────────┘              │ + emotion tags   │   (+ fine-tuned    └────────────┘
                             │                  │     checkpoints)
                             └──────────────────┘
```

## Train a custom model in three commands

No ML setup, no dataset wrangling, no config files. The whole flow is:

1. **Add audio** — point `add-sample` at any clips of the person; they're
   auto-transcribed and normalized for you:
   ```bash
   uv run voiceclone add-sample alice clip1.mp3 clip2.wav --emotion neutral
   ```
2. **Train** — one command builds the dataset and fine-tunes a dedicated model
   on those samples (engine-specific recipe, run automatically):
   ```bash
   uv run voiceclone train alice --engine cosyvoice3     # official CosyVoice 3 recipe
   uv run voiceclone train alice --engine xtts-v2 --epochs 1 --lr 4e-6   # Coqui recipe
   # add --dry-run to just prep the dataset
   ```
3. **Use it** — switch synthesis to the custom model:
   ```bash
   uv run voiceclone synthesize "Hi!" --voice alice --engine cosyvoice3 --mode finetuned
   ```

That's it — transcription, sentence splitting, and checkpoint management are all
handled for you. Each engine keeps its own per-voice checkpoints, so a voice can
be fine-tuned with several engines side by side.

## Features

- **Zero-shot cloning** — drop in 10–60 s of clean speech, transcribe with
  Whisper, synthesize immediately. No training required.
- **Multi-engine** — XTTS v2 (legacy), CosyVoice 3 and Chatterbox V3 behind one
  interface; pick per call (`--engine`) or set a default. See [Engines](#engines).
- **Sentiment control** — tag samples with emotions (`happy`, `sad`, …) or pass
  free-text style (`"whisper this, very calm"`). At synthesis time the toolkit
  picks the voice's best-matching sample as the reference clip; the engine
  carries that prosody/emotion into the output (Chatterbox additionally maps
  emotions to its intensity dial).
- **Optional fine-tuning** — train a dedicated per-voice model for higher
  fidelity. CosyVoice 3 runs FunAudioLLM's official recipe end-to-end; XTTS v2
  uses Coqui's GPT trainer. Works on CPU, much faster on GPU.
- **CLI + web UI** — scriptable commands and a local browser app for upload,
  tagging, synthesis and training (with engine selection).
- **Wide language coverage** — English + Dutch first-class; Chatterbox V3 adds
  21 more languages zero-shot, CosyVoice 3 covers 9 (no Dutch).

## Quickstart

```bash
# 1) install (Python 3.11 venv is created automatically by uv)
cd voiceclone
uv sync

# 2) one-time setup: accept license + download models (~2.4 GB)
uv run voiceclone init --accept-license --download

# 3) register a voice from plain audio files (auto-transcribed)
uv run voiceclone add-sample alice clip1.mp3 clip2.wav --emotion neutral
uv run voiceclone add-sample alice sad_clip.wav --emotion sad
uv run voiceclone voices            # see what you have

# 4) synthesize with sentiment (default engine; see `engines` for options)
uv run voiceclone synthesize "Hello, this is a cloned voice!" --voice alice
uv run voiceclone synthesize "Ik ben blij om je te zien." --voice alice --lang nl
uv run voiceclone synthesize "Goodbye." --voice alice --emotion sad
uv run voiceclone synthesize "Shh…" --voice alice --style "whisper, very quiet"

# 5) optional: install another engine (dedicated venv + weights) and use it
uv run voiceclone engines                      # what's available / installed
uv run voiceclone install-engine chatterbox    # ~4 GB venv; weights on first use
uv run voiceclone synthesize "Hallo!" --voice alice --engine chatterbox

# 6) optional: fine-tune a dedicated model (slow on CPU!)
uv run voiceclone train alice --engine xtts-v2 --epochs 1 --lr 4e-6   # add --dry-run to only prep data
uv run voiceclone synthesize "Hi!" --voice alice --mode finetuned

# web UI
uv run voiceclone serve --port 8321                 # http://127.0.0.1:8321
```

## Engines

| Engine | License (code / weights) | Zero-shot languages | Fine-tune | Notes |
| --- | --- | --- | --- | --- |
| `xtts-v2` | Coqui Public Model / **CPML (non-commercial)** | 17 incl. en, nl | ✔ Coqui GPT recipe | Legacy default; kept for backward compatibility. Upstream unmaintained. |
| `cosyvoice3` | **Apache-2.0 / Apache-2.0** | 9 (zh/en/ja/ko/de/es/fr/it/ru — no nl) | ✔ official FunAudioLLM recipe | Best "import → auto-train" path; 0.5B model, ~6 GB weights. |
| `chatterbox` | **MIT / MIT** | 23 incl. en, nl | ✘ (no official recipe yet) | Resemble AI Multilingual V3; emotion → intensity dial; outputs carry an imperceptible PerTh watermark. |

External engines (`cosyvoice3`, `chatterbox`) pin dependency versions that
conflict with the toolkit's own environment, so each runs in a **dedicated venv**
under `data/engines/<name>/` and is driven through a small JSON-lines worker
process (model loaded once, kept warm between calls). Install them with:

```bash
uv run voiceclone install-engine cosyvoice3     # git clone + py3.10 venv + deps (~7 GB)
uv run voiceclone install-engine chatterbox     # pip chatterbox-tts into a py3.10 venv (~4 GB)
# CosyVoice weights (~6 GB) download automatically on first use (or: voiceclone init --download)
```

Installs are idempotent — re-running prints "Already installed" after re-checking
the torch pins and patches (no upstream update check; engine versions are pinned
on purpose). To force the full install path again (re-apply pins/patches,
reinstall the pinned package into the existing venv): `uv run voiceclone
install-engine <name> --force`.

**Disk usage & caches.** Everything the toolkit writes lives under the project's
`data/` directory — including package caches (`data/cache/uv`, `data/cache/pip`,
`data/cache/hf`) and temp files (`data/tmp`). Installs therefore land on the drive
where voiceclone sits, not on your home partition. The caches are shared between
engines (e.g. torch wheels are stored once) and safe to delete at any time
(`rm -rf data/cache` — they will be re-downloaded if needed). If you also want
`uv` itself (outside `voiceclone`) to use the project cache, set
`export UV_CACHE_DIR=$PWD/data/cache/uv` in your shell.

**Considered, not integrated:** F5-TTS (excellent code, but pretrained weights
are CC-BY-NC and officially zh/en only), GPT-SoVITS (MIT + strong 1-minute
few-shot pipeline, but git-clone-only packaging with heavy sys.path hacks and
no Dutch), Spark-TTS (CC-BY-NC-SA, dormant, no training code), Higgs-Audio-v2
(archived, ≥24 GB VRAM). Revisit as these projects evolve — the engine
interface (`voiceclone/engines/base.py`) is designed for exactly this.

Set the default engine in `data/settings.json`: `{"default_engine": "cosyvoice3"}`.

## How sentiment works

XTTS v2 has no explicit "emotion" parameter — emotional tone is largely carried
by the **reference clip**. This toolkit turns that into a practical control:

1. When you register samples, each gets an emotion tag (default `neutral`).
   Tag clips with `--emotion sad` / in the web UI when they sound sad/happy/etc.
2. Free-text styles are mapped to presets by keywords
   (`"angry and loud"` → `angry`, `"whisper, calm"` → `calm`).
3. At synthesis time the best-matching tagged sample is chosen as reference
   (duration sweet-spot 3–15 s, deterministic). If no matching tag exists it
   falls back to your closest available sample — so plain untagged samples
   still work, they just all sound "neutral".

**Tip:** for reliable sentiment, record/tag a few clips of the person *actually*
speaking in each emotion you want.

(Engine nuance: XTTS v2 and CosyVoice 3 carry emotion through the reference
clip; Chatterbox additionally maps the chosen emotion onto its "exaggeration"
intensity dial — `neutral`≈0.5, `calm`≈0.35, `happy/excited/angry`≈0.7–0.8.)

## Fine-tuning notes

- **Shared dataset prep** (all engines): samples are split into sentence-level
  clips (Whisper word timestamps) and written as `wavs/ + metadata_train.csv /
  metadata_eval.csv`. Bilingual voices get one dataset per language.
- **`--engine cosyvoice3`** runs FunAudioLLM's official CosyVoice 3 recipe
  headlessly: Kaldi-style data → parquet → `torchrun train.py` on the LLM
  component (where speaker identity lives) → averaged best checkpoint → an
  inference-ready model dir. Trains only the LLM by design; flow/vocoder stay
  base and adapt via the prompt at inference time.
- **`--engine xtts-v2`** follows Coqui's official XTTS GPT fine-tune recipe:
  train from the original XTTS v2 weights with `metadata_train/eval.csv`.
- **CPU:** works, but expect *hours* for a few minutes of audio
  (`batch_size=1`, gradient accumulation). Use `--dry-run` to prepare data only.
- **RAM:** fine-tuning needs ~16 GiB free (fp32 model + gradients + optimizer
  states). `train` refuses to start below 12 GiB of free RAM unless you pass
  `--force`. Prefer a GPU machine or one with ≥ 16 GiB RAM.
- **GPU:** same command on a CUDA machine → minutes to ~1 hour. When a CUDA
  GPU is present, training automatically uses bfloat16 mixed precision
  (~2× faster, lower memory; master weights stay fp32). Override with
  `--precision fp32` (or force `--precision bf16`).
- Quality guidance: ≥ 1–5 min of clean, single-speaker audio; avoid music,
  noise and long silences in the source clips.

## Project layout

```
voiceclone/
├── cli.py            # CLI (init/add-sample/voices/tag/synthesize/train/engines/install-engine/serve)
├── config.py         # settings + data dir layout
├── audio.py          # decode/trim/normalize via PyAV (no system ffmpeg needed)
├── transcribe.py     # faster-whisper transcription + sentence splitting
├── emotion.py        # preset emotions, style→emotion mapping, reference pick
├── voices.py         # voice profile store (JSON + WAV under data/voices/)
├── speaker.py        # optional ECAPA speaker-consistency checks
├── synthesize.py     # orchestration: emotion → reference → engine
├── train.py          # fine-tune dispatcher (per-engine recipes)
├── download.py       # streaming model downloader
├── server.py         # FastAPI backend for the web UI
├── engines/
│   ├── __init__.py   # engine registry (specs, install detection, selection)
│   ├── base.py       # engine interface
│   ├── xtts.py       # XTTS v2 engine (in-process)
│   ├── external.py   # subprocess-isolated engine base + JSON-lines worker protocol
│   ├── cosyvoice.py  # CosyVoice 3 (install, official finetune recipe, worker client)
│   ├── chatterbox.py # Chatterbox Multilingual V3 (install, worker client)
│   └── workers/      # worker scripts that run INSIDE the engines' own venvs
└── web/              # static frontend (vanilla JS, no build step)
```

All user data lives in `data/` (override with `VOICECLONE_DATA`):

```
data/
├── voices/<name>/voice.json      # metadata, samples, emotion tags, per-engine finetune refs
├── voices/<name>/samples/*.wav   # your normalized 24 kHz clips
├── models/                       # base weights + fine-tuned checkpoints (per engine)
├── engines/<name>/               # external engines: venv/, repo/ (CV3), models/, worker.log
├── output/                       # generated speech
└── logs/                         # training logs
```

## Using a GPU

The 5090 (Blackwell) needs a recent driver (≥ ~580, CUDA 13 capable).
Training auto-enables bfloat16 mixed precision when the GPU is visible.

## License & legal notes

- **XTTS v2 weights** are under the **Coqui Public Model License — non-commercial
  use only**. Acceptance is required once (`voiceclone init --accept-license`)
  and recorded in `data/models/LICENSES/`. For commercial use, pick a
  permissive engine instead: **CosyVoice 3 (Apache-2.0)** or **Chatterbox (MIT)**.
- Whisper weights: MIT. This toolkit's code: use as you like; provided as-is.
- Chatterbox outputs carry Resemble AI's imperceptible PerTh neural watermark
  (responsible-AI measure; not a usage restriction — the license is MIT).
- **You are responsible** for having rights to the voice samples you clone and
  for complying with local laws on voice cloning/consent.

## Roadmap / ideas

- [ ] Speaker-consistency warnings in the UI (ECAPA embeddings, module exists)
- [ ] Chatterbox fine-tune support when an official recipe appears (community
      HF-Trainer recipes exist today)
- [ ] Re-evaluate GPT-SoVITS / F5-TTS integration (see "Considered" above)
- [ ] Batch synthesis (script → multiple wavs), streaming API
- [ ] Voice quality scoring (SNR, clipping, length distribution) on upload
