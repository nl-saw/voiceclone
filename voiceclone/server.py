"""FastAPI backend for the web UI.

Endpoints (all JSON unless noted):
  GET  /api/voices                     list voices with summary stats
  GET  /api/voices/{name}              voice detail (samples, tags, finetune info)
  POST /api/voices/{name}/samples      multipart upload of .wav/.mp3 files (+ optional lang/emotion per batch)
  PATCH /api/voices/{name}/samples/{id}   {emotion?, note?}
  DELETE /api/voices/{name}/samples/{id}
  GET  /api/emotions                   preset emotion list
  POST /api/synthesize                 {voice, text, emotion?, style?, language?, mode?} -> wav bytes + meta
  POST /api/train                      {voice, epochs?, batch_size?, grad_accum_steps?, precision?, lr?, force?, dry_run?} -> job id (+advisory)
  GET  /api/train/{job_id}             job status (running/done/failed) + tail of log
  GET  /api/storage                    disk-usage breakdown + all fine-tune runs
  POST /api/storage/clean              {voice, action: run|all-but-registered|reset, run?, force?} -> freed bytes
  GET  /api/voices/{name}/checkpoints  list a voice's run dirs (sizes, steps, which is registered)
  POST /api/voices/{name}/checkpoint   {checkpoint: path} -> register that checkpoint
  DELETE /api/voices/{name}/checkpoint clear registration (fall back to zero-shot)
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from .config import get_settings
from .emotion import PRESET_EMOTIONS
from .storage import (
    RECOMMENDED_MIN_AUDIO_S,
    clean_voice,
    clear_checkpoint,
    list_runs,
    register_checkpoint,
    scan_storage,
)
from .voices import (
    VoiceError,
    add_samples,
    list_voices,
    load_voice,
    remove_sample,
    tag_sample,
)


class SynthesizeRequest(BaseModel):
    voice: str
    text: str
    emotion: str | None = None
    style: str | None = None
    language: str | None = "auto"
    mode: str = "auto"  # auto | zero-shot | finetuned
    # Optional generation overrides (None = model config default).
    temperature: float | None = None
    length_penalty: float | None = None
    repetition_penalty: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    speed: float | None = None
    max_chars: int | None = None  # long-text chunk cap (None = engine default 120, 0 = off)


class TrainRequest(BaseModel):
    voice: str
    epochs: int = 5
    batch_size: int = 1
    grad_accum_steps: int = 4
    precision: str = "auto"  # auto | bf16 | fp32
    lr: float | None = None  # default 4e-06; lower = stays closer to base voice
    force: bool = False      # start even if free RAM looks too low
    dry_run: bool = False


class CleanRequest(BaseModel):
    voice: str
    action: str  # run | all-but-registered | reset
    run: str | None = None  # run dir name (only for action="run")
    force: bool = False


class CheckpointRequest(BaseModel):
    checkpoint: str  # absolute path to an existing best_model_*.pth


# In-memory job registry (server process lifetime).
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="voiceclone", version="0.1.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    web_dir = Path(__file__).resolve().parent.parent / "web"

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(web_dir / "index.html")

    @app.get("/style.css")
    def style_css() -> FileResponse:
        return FileResponse(web_dir / "style.css")

    @app.get("/app.js")
    def app_js() -> FileResponse:
        return FileResponse(web_dir / "app.js")

    # ------------------------------------------------------------------ #
    @app.get("/api/emotions")
    def emotions() -> list[str]:
        return PRESET_EMOTIONS

    @app.get("/api/voices")
    def voices_list() -> list[dict]:
        out = []
        for v in list_voices():
            out.append({
                "name": v.name,
                "samples": len(v.samples),
                "total_seconds": round(v.total_seconds, 1),
                "languages": sorted({s.language for s in v.samples}),
                "emotions": sorted({s.emotion for s in v.samples if s.emotion != "neutral"}),
                "finetuned": bool(v.finetuned),
            })
        return out

    @app.get("/api/voices/{name}")
    def voice_detail(name: str) -> dict:
        try:
            v = load_voice(name)
        except VoiceError as e:
            raise HTTPException(404, str(e)) from e
        return {
            "name": v.name,
            "created_at": v.created_at,
            "total_seconds": round(v.total_seconds, 1),
            "finetuned": v.finetuned,
            "samples": [s.to_dict() for s in v.samples],
        }

    @app.post("/api/voices/{name}/samples")
    async def upload_samples(
        name: str,
        files: list[UploadFile] = File(...),
        lang: str = Form("auto"),
        emotion: str = Form("neutral"),
        note: str = Form(""),
    ) -> dict:
        tmpdir = Path(tempfile.mkdtemp(prefix="vc_upload_"))
        paths: list[str] = []
        try:
            for i, f in enumerate(files):
                suffix = Path(f.filename or "upload.bin").suffix or ".wav"
                dest = tmpdir / f"{i:02d}{suffix}"
                with open(dest, "wb") as out:
                    shutil.copyfileobj(f.file, out)
                paths.append(str(dest))
            v, reports = add_samples(
                name, paths,
                language=None if lang == "auto" else lang,
                emotion=emotion, note=note,
            )
        except VoiceError as e:
            raise HTTPException(400, str(e)) from e
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return {
            "voice": v.name,
            "samples_total": len(v.samples),
            "reports": reports,
        }

    @app.patch("/api/voices/{name}/samples/{sid}")
    def patch_sample(name: str, sid: str, payload: dict) -> dict:
        try:
            v = tag_sample(
                name, sid,
                emotion=payload.get("emotion"),
                note=payload.get("note"),
            )
        except VoiceError as e:
            raise HTTPException(400, str(e)) from e
        s = next(x for x in v.samples if x.id == sid)
        return s.to_dict()

    @app.delete("/api/voices/{name}/samples/{sid}")
    def delete_sample(name: str, sid: str) -> dict:
        try:
            v = remove_sample(name, sid)
        except VoiceError as e:
            raise HTTPException(400, str(e)) from e
        return {"voice": v.name, "samples_total": len(v.samples)}

    # ------------------------------------------------------------------ #
    @app.post("/api/synthesize")
    def synthesize(req: SynthesizeRequest) -> Response:
        try:
            voice = load_voice(req.voice)
        except VoiceError as e:
            raise HTTPException(404, str(e)) from e

        from .synthesize import synthesize as do_synthesize

        t0 = time.time()
        try:
            outcome = do_synthesize(
                voice=voice,
                text=req.text,
                emotion=req.emotion,
                style=req.style,
                language=None if req.language in (None, "auto") else req.language,
                engine_mode=req.mode,
                temperature=req.temperature,
                length_penalty=req.length_penalty,
                repetition_penalty=req.repetition_penalty,
                top_k=req.top_k,
                top_p=req.top_p,
                speed=req.speed,
                max_chars=req.max_chars,
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"{type(e).__name__}: {e}") from e

        meta = {
            "engine": outcome.result.engine,
            "mode": outcome.result.mode,
            "requested_emotion": outcome.requested_emotion,
            "reference_emotion": outcome.resolved_emotion,
            "reference_file": Path(outcome.result.reference_file).name,
            "duration_s": round(len(outcome.result.wav) / outcome.result.sample_rate, 2),
            "elapsed_s": round(time.time() - t0, 1),
            "output_path": str(outcome.output_path),
        }
        wav_bytes = _to_wav_bytes(outcome.result.wav, outcome.result.sample_rate)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"X-Synthesis-Meta": _json_header(meta)},
        )

    # ------------------------------------------------------------------ #
    @app.post("/api/train")
    def train(req: TrainRequest) -> dict:
        try:
            v = load_voice(req.voice)
        except VoiceError as e:
            raise HTTPException(404, str(e)) from e
        if not v.samples:
            raise HTTPException(400, "Voice has no samples.")

        job_id = uuid.uuid4().hex[:12]
        settings = get_settings()
        log_path = settings.data_dir / "logs" / f"train_{v.name}_{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with _jobs_lock:
            _jobs[job_id] = {
                "voice": v.name,
                "status": "running",
                "started_at": time.time(),
                "log_path": str(log_path),
                "error": None,
                "checkpoint": None,
            }

        def runner() -> None:
            from .train import record_finetune, run_finetune

            try:
                report = run_finetune(
                    v.name,
                    epochs=req.epochs,
                    batch_size=req.batch_size,
                    grad_accum_steps=req.grad_accum_steps,
                    precision=req.precision,
                    lr=req.lr,
                    force=req.force,
                    dry_run=req.dry_run,
                    log_path=log_path,
                )
                if not req.dry_run and report.checkpoint:
                    record_finetune(v.name, report)
                with _jobs_lock:
                    job = _jobs[job_id]
                    job["status"] = "done"
                    job["finished_at"] = time.time()
                    job["checkpoint"] = report.checkpoint
            except Exception as e:  # noqa: BLE001
                with _jobs_lock:
                    job = _jobs[job_id]
                    job["status"] = "failed"
                    job["error"] = f"{type(e).__name__}: {e}"

        # Advisory: below ~10 min of audio, fine-tuning usually *hurts* word
        # accuracy (it perturbs the pretrained text->speech mapping faster than
        # it learns voice traits). Surface this so the user makes an informed call.
        advisory = None
        if v.total_seconds < RECOMMENDED_MIN_AUDIO_S:
            advisory = (
                f"Only {v.total_seconds:.0f}s of source audio (recommended ≥ "
                f"{RECOMMENDED_MIN_AUDIO_S // 60} min). With this little data, more epochs "
                f"tend to make words WORSE, not better — 1 epoch is usually the ceiling. "
                f"Zero-shot often sounds cleaner; collect more audio for real fine-tune gains."
            )

        threading.Thread(target=runner, daemon=True).start()
        return {"job_id": job_id, "advisory": advisory}

    @app.get("/api/train/{job_id}")
    def train_status(job_id: str) -> dict:
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job id")
        out = dict(job)
        log_path = Path(job["log_path"])
        if log_path.exists():
            tail = log_path.read_text(errors="replace").splitlines()[-12:]
            out["log_tail"] = tail
        return out

    # ------------------------------------------------------------------ #
    # storage & cleanup
    # ------------------------------------------------------------------ #
    @app.get("/api/storage")
    def storage() -> dict:
        return scan_storage()

    @app.post("/api/storage/clean")
    def storage_clean(req: CleanRequest) -> dict:
        try:
            return clean_voice(
                req.voice, action=req.action, run=req.run, force=req.force
            )
        except VoiceError as e:
            raise HTTPException(400, str(e)) from e

    # ------------------------------------------------------------------ #
    # checkpoint management (per voice)
    # ------------------------------------------------------------------ #
    @app.get("/api/voices/{name}/checkpoints")
    def checkpoints_list(name: str) -> dict:
        try:
            return list_runs(name)
        except VoiceError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/voices/{name}/checkpoint")
    def checkpoint_register(name: str, req: CheckpointRequest) -> dict:
        try:
            return register_checkpoint(name, req.checkpoint)
        except VoiceError as e:
            raise HTTPException(400, str(e)) from e

    @app.delete("/api/voices/{name}/checkpoint")
    def checkpoint_clear(name: str) -> dict:
        try:
            return clear_checkpoint(name)
        except VoiceError as e:
            raise HTTPException(404, str(e)) from e

    return app


def _to_wav_bytes(wav, sr: int) -> bytes:
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, wav.astype("float32"), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _json_header(obj) -> str:
    import base64
    import json

    return base64.b64encode(json.dumps(obj).encode()).decode()
