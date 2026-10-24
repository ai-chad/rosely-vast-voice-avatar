"""Rosely Ditto talking-head renderer (TensorRT, RTX 3090). Opt-in alongside MuseTalk.

Ditto animates the whole portrait (head motion, blinks, expressions) instead of
only the mouth, and accepts per-reply emotion + gesture controls (acting.py).
The model is held resident (~2.6 GB VRAM); one render at a time.
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

import acting

LOG = logging.getLogger("uvicorn.error")  # uvicorn configures this one; custom loggers stay silent
BUNDLE_ROOT = Path(os.getenv("BUNDLE_ROOT", "/workspace/rosely-voice-musetalk-v1"))
DITTO_ROOT = Path(os.getenv("DITTO_ROOT", str(BUNDLE_ROOT / "source" / "ditto-talkinghead")))
DITTO_MODELS = Path(os.getenv("DITTO_MODELS", str(BUNDLE_ROOT / "models" / "ditto")))
API_KEY = os.environ.get("VOICE_API_KEY", "")
MAX_IMAGE_BYTES = int(os.getenv("MAX_AVATAR_SOURCE_BYTES", str(25 * 1024 * 1024)))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AVATAR_AUDIO_BYTES", str(10 * 1024 * 1024)))
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AVATAR_AUDIO_SECONDS", "12"))
render_lock = asyncio.Lock()
sdk = None


def load_sdk():
    sys.path.insert(0, str(DITTO_ROOT))
    os.chdir(DITTO_ROOT)  # upstream modules resolve some helpers relative to the repo
    from stream_pipeline_offline import StreamSDK
    return StreamSDK(str(DITTO_MODELS / "ditto_cfg" / "v0.4_hubert_cfg_trt.pkl"),
                     str(DITTO_MODELS / "ditto_trt_Ampere_Plus"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sdk
    started = time.monotonic()
    sdk = await asyncio.to_thread(load_sdk)
    LOG.info("Ditto TensorRT models loaded in %.1fs", time.monotonic() - started)
    yield


app = FastAPI(title="Rosely Ditto Renderer", version="0.1.0", lifespan=lifespan)


def authorize(key: str | None) -> None:
    if not API_KEY or not key or not hmac.compare_digest(key, API_KEY):
        raise HTTPException(401, "invalid API key")


@app.get("/health")
async def health():
    return {"ok": sdk is not None, "service": "ditto", "engine": "tensorrt", "busy": render_lock.locked()}


def render(image_path: Path, audio_path: Path, out_path: Path, emotion: str, gesture: str) -> dict:
    import librosa  # deferred: heavy import only in the render thread
    audio, _ = librosa.load(str(audio_path), sr=16000)
    seconds = len(audio) / 16000
    if not 0 < seconds <= MAX_AUDIO_SECONDS:
        raise HTTPException(413, f"audio must be between 0 and {MAX_AUDIO_SECONDS:g} seconds")
    num_frames = math.ceil(seconds * acting.FPS)
    labels, ctrl = acting.controls(emotion, gesture, num_frames)
    silent = out_path.with_suffix(".silent.mp4")
    started = time.monotonic()
    sdk.setup(str(image_path), str(silent), emo=labels)
    sdk.setup_Nd(N_d=num_frames, ctrl_info=ctrl)
    sdk.audio2motion_queue.put(sdk.wav2feat.wav2feat(audio))
    sdk.close()
    took = time.monotonic() - started
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", sdk.tmp_output_path, "-i", str(audio_path),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)],
                   check=True, timeout=120)
    return {"seconds": round(seconds, 2), "render_seconds": round(took, 2), "frames": num_frames}


@app.post("/v1/ditto/render")
async def render_clip(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    emotion: str = Form("neutral"),
    gesture: str = Form("none"),
    x_api_key: str | None = Header(default=None),
):
    authorize(x_api_key)
    if sdk is None:
        raise HTTPException(503, "Ditto models are still loading")
    suffix = Path(image.filename or "character.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(400, "Ditto needs a still portrait (png, jpg or webp)")
    image_bytes, audio_bytes = await image.read(MAX_IMAGE_BYTES + 1), await audio.read(MAX_AUDIO_BYTES + 1)
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES or not audio_bytes or len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "empty or oversized upload")
    if render_lock.locked():
        raise HTTPException(429, "Ditto renderer busy; retry")
    async with render_lock:
        with tempfile.TemporaryDirectory(prefix="ditto-") as tmp:
            work = Path(tmp)
            (work / f"character{suffix}").write_bytes(image_bytes)
            (work / "speech.wav").write_bytes(audio_bytes)
            try:
                info = await asyncio.to_thread(render, work / f"character{suffix}", work / "speech.wav",
                                               work / "reply.mp4", emotion, gesture)
            except HTTPException:
                raise
            except Exception as exc:
                LOG.exception("Ditto render failed")
                raise HTTPException(500, f"Ditto render failed: {type(exc).__name__}") from None
            LOG.info("ditto_render emotion=%s gesture=%s %s", emotion, gesture, info)
            return Response((work / "reply.mp4").read_bytes(), media_type="video/mp4",
                            headers={"X-Render-Seconds": str(info["render_seconds"])})
