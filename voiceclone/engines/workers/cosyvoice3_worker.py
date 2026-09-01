"""CosyVoice 3 worker — runs INSIDE the engine's dedicated venv (not ours).

Speaks the voiceclone external-engine protocol (JSON lines on stdin/stdout):

  {"op": "ping"}                          -> {"ok": true, ...}
  {"op": "synthesize", "text", "reference_wav_path", "reference_text",
   "language", "emotion", "style", "finetuned_checkpoint"}
                                          -> {"ok": true, "wav_path", "sample_rate", "mode"}

The model is loaded lazily on the first synthesis and kept warm. A different
``finetuned_checkpoint`` (a CosyVoice model dir with a trained llm.pt) triggers
a reload. Configuration comes from environment variables set by the parent:

  VC_ENGINE_DIR   per-engine state dir (repo/, models/, tmp/)
  VC_COSY_MODEL_DIR   base Fun-CosyVoice3-0.5B model dir
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def send(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def main() -> None:
    engine_dir = Path(os.environ["VC_ENGINE_DIR"])
    repo = engine_dir / "repo"
    model_dir = Path(os.environ.get("VC_COSY_MODEL_DIR", str(engine_dir / "models" / "Fun-CosyVoice3-0.5B")))

    # CosyVoice is a git checkout, not a pip package; Matcha-TTS is its submodule.
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "third_party" / "Matcha-TTS"))

    (engine_dir / "tmp").mkdir(parents=True, exist_ok=True)

    state: dict = {"model": None, "model_for": None}

    def get_model(finetuned: str | None):
        import torch  # noqa: F401 — makes OOM/missing-CUDA errors surface here

        key = finetuned or "__base__"
        if state["model"] is None or state["model_for"] != key:
            from cosyvoice.cli.cosyvoice import AutoModel

            t0 = time.time()
            state["model"] = AutoModel(model_dir=finetuned or str(model_dir))
            state["model_for"] = key
            print(f"[cosyvoice3] loaded model {key} in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
        return state["model"]

    send({"ok": True, "op": "ready", "engine": "cosyvoice3"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
            op = cmd.get("op")

            if op == "ping":
                send({"ok": True, "engine": "cosyvoice3", "model_loaded": state["model"] is not None})

            elif op == "synthesize":
                t0 = time.time()
                finetuned = cmd.get("finetuned_checkpoint") or None
                m = get_model(finetuned)

                import torchaudio

                prompt_text = (cmd.get("reference_text") or "").strip()
                # Fun-CosyVoice3 requires the instruct prefix on the prompt text.
                if not prompt_text.startswith("You are a helpful assistant.<|endofprompt|>"):
                    prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt_text

                chunks = []
                for chunk in m.inference_zero_shot(
                    cmd["text"], prompt_text, cmd["reference_wav_path"], stream=False
                ):
                    chunks.append(chunk["tts_speech"])  # (1, N) tensor
                wav = chunks[0] if len(chunks) == 1 else __import__("torch").cat(chunks, dim=-1)

                out_path = Path(engine_dir / "tmp" / f"out_{int(time.time() * 1000)}.wav")
                torchaudio.save(str(out_path), wav, m.sample_rate)
                send({
                    "ok": True,
                    "wav_path": str(out_path),
                    "sample_rate": int(m.sample_rate),
                    "mode": "finetuned" if finetuned else "zero-shot",
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
