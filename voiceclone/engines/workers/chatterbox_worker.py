"""Chatterbox Multilingual V3 worker — runs INSIDE the engine's dedicated venv.

Speaks the voiceclone external-engine protocol (JSON lines on stdin/stdout).
Model: ResembleAI Chatterbox Multilingual V3 (0.5B, 23 languages incl. Dutch,
MIT license). Loaded lazily on first synthesis and kept warm.

Environment variables (set by the parent process):
  VC_ENGINE_DIR   per-engine state dir (venv/, tmp/, hf/ cache)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# voiceclone emotion tags -> Chatterbox "exaggeration" intensity dial [0..1]
EMOTION_EXAGGERATION = {
    "neutral": 0.5,
    "calm": 0.35,
    "sad": 0.45,
    "fear": 0.6,
    "surprise": 0.65,
    "happy": 0.7,
    "excited": 0.8,
    "angry": 0.8,
}


def send(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def main() -> None:
    engine_dir = Path(os.environ["VC_ENGINE_DIR"])
    (engine_dir / "tmp").mkdir(parents=True, exist_ok=True)
    state: dict = {"model": None}

    def get_model():
        if state["model"] is None:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            device = "cuda" if torch.cuda.is_available() else "cpu"
            t0 = time.time()
            state["model"] = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
            print(f"[chatterbox] loaded V3 on {device} in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
        return state["model"]

    send({"ok": True, "op": "ready", "engine": "chatterbox"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
            op = cmd.get("op")

            if op == "ping":
                send({"ok": True, "engine": "chatterbox", "model_loaded": state["model"] is not None})

            elif op == "synthesize":
                t0 = time.time()
                m = get_model()
                import torchaudio

                lang = (cmd.get("language") or "en").lower()
                emotion = (cmd.get("emotion") or "neutral").lower()
                wav = m.generate(
                    cmd["text"],
                    language_id=lang,
                    audio_prompt_path=cmd.get("reference_wav_path"),
                    exaggeration=EMOTION_EXAGGERATION.get(emotion, 0.5),
                )
                out_path = Path(engine_dir / "tmp" / f"out_{int(time.time() * 1000)}.wav")
                torchaudio.save(str(out_path), wav.detach().cpu(), m.sr)
                send({
                    "ok": True,
                    "wav_path": str(out_path),
                    "sample_rate": int(m.sr),
                    "mode": "zero-shot",
                    "elapsed_s": round(time.time() - t0, 1),
                })

            else:
                send({"ok": False, "error": f"unknown op {op!r}"})

        except Exception as e:  # noqa: BLE001 — report any failure back to the parent
            import traceback

            traceback.print_exc()
            send({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
