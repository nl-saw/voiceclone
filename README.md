# voiceclone 🎙️

A local **voice cloning toolkit**: register a person's voice from plain audio
files (`.wav`/`.mp3`/…), optionally fine-tune a dedicated model per voice, and
synthesize speech **with sentiment control** (happy / sad / angry / calm / …) —
in **English and Dutch**, fully offline after setup.

Built on [XTTS v2](https://huggingface.co/coqui/XTTS-v2) (Coqui), which natively
supports 17 languages including English (`en`) and Dutch (`nl`).

```
┌─────────────┐   samples    ┌──────────────────┐   text + emotion   ┌────────────┐
│ .wav/.mp3   │─────────────▶│ voice profile    │───────────────────▶│  speech    │
│ (any length)│              │ + transcripts    │   XTTS v2 engine   │  .wav out  │
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
2. **Train** — one command builds the dataset and fine-tunes a dedicated
   XTTS v2 model on those samples:
   ```bash
   uv run voiceclone train alice --epochs 1        # --dry-run to just prep data
   ```
3. **Use it** — switch synthesis to the custom model:
   ```bash
   uv run voiceclone synthesize "Hi!" --voice alice --mode finetuned
   ```

That's it — transcription, sentence splitting, and checkpoint management are all
handled for you.

## Features

- **Zero-shot cloning** — drop in 10–60 s of clean speech, transcribe with
  Whisper, synthesize immediately. No training required.
- **Sentiment control** — tag samples with emotions (`happy`, `sad`, …) or pass
  free-text style (`"whisper this, very calm"`). At synthesis time the toolkit
  picks the voice's best-matching sample as the reference clip; XTTS v2 carries
  that prosody/emotion into the output.
- **Optional fine-tuning** — train a dedicated XTTS v2 model on a voice's
  samples for higher fidelity (works on CPU, much faster on GPU).
- **CLI + web UI** — scriptable commands and a local browser app for upload,
  tagging, synthesis and training.
- **English + Dutch** first-class; Whisper auto-transcribes both.

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

# 4) synthesize with sentiment
uv run voiceclone synthesize "Hello, this is a cloned voice!" --voice alice
uv run voiceclone synthesize "Ik ben blij om je te zien." --voice alice --lang nl
uv run voiceclone synthesize "Goodbye." --voice alice --emotion sad
uv run voiceclone synthesize "Shh…" --voice alice --style "whisper, very quiet"

# 5) optional: fine-tune a dedicated model (slow on CPU!)
uv run voiceclone train alice --epochs 1            # add --dry-run to only prep data
uv run voiceclone synthesize "Hi!" --voice alice --mode finetuned

# web UI
uv run voiceclone serve --port 8321                 # http://127.0.0.1:8321
```

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

## Fine-tuning notes

- The pipeline follows Coqui's official XTTS GPT fine-tune recipe: samples are
  split into sentence-level clips (Whisper word timestamps), written as
  `wavs/ + metadata_train.csv / metadata_eval.csv`, then trained from the
  original XTTS v2 weights. Bilingual voices get one dataset per language.
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
├── cli.py            # CLI (init/add-sample/voices/tag/synthesize/train/serve)
├── config.py         # settings + data dir layout
├── audio.py          # decode/trim/normalize via PyAV (no system ffmpeg needed)
├── transcribe.py     # faster-whisper transcription + sentence splitting
├── emotion.py        # preset emotions, style→emotion mapping, reference pick
├── voices.py         # voice profile store (JSON + WAV under data/voices/)
├── speaker.py        # optional ECAPA speaker-consistency checks
├── synthesize.py     # orchestration: emotion → reference → engine
├── train.py          # fine-tuning pipeline (official XTTS GPT trainer)
├── download.py       # streaming model downloader
├── server.py         # FastAPI backend for the web UI
├── engines/
│   ├── base.py       # engine interface (XTTS + future engines)
│   └── xtts.py       # XTTS v2 engine (zero-shot + fine-tuned checkpoints)
└── web/              # static frontend (vanilla JS, no build step)
```

All user data lives in `data/` (override with `VOICECLONE_DATA`):

```
data/
├── voices/<name>/voice.json      # metadata, samples, emotion tags, finetune ref
├── voices/<name>/samples/*.wav   # your normalized 24 kHz clips
├── models/                       # XTTS v2 weights + fine-tuned checkpoints
├── output/                       # generated speech
└── logs/                         # training logs
```

## Using a GPU

The 5090 (Blackwell) needs a recent driver (≥ ~580, CUDA 13 capable).
Training auto-enables bfloat16 mixed precision when the GPU is visible.

## License & legal notes

- **XTTS v2 weights** are under the **Coqui Public Model License — non-commercial
  use only**. Acceptance is required once (`voiceclone init --accept-license`)
  and recorded in `data/models/LICENSES/`. If you need commercial rights, look at
  permissively licensed engines (e.g. CosyVoice 2, Apache-2.0) — the engine
  interface here makes that a contained change.
- Whisper weights: MIT. This toolkit's code: use as you like; provided as-is.
- **You are responsible** for having rights to the voice samples you clone and
  for complying with local laws on voice cloning/consent.

## Roadmap / ideas

- [ ] Speaker-consistency warnings in the UI (ECAPA embeddings, module exists)
- [ ] Instruction-based engine slot (e.g. CosyVoice 2 / OpenAudio S1-mini) for
      direct "speak sadly" control on top of reference selection
- [ ] Batch synthesis (script → multiple wavs), streaming API
- [ ] Voice quality scoring (SNR, clipping, length distribution) on upload
