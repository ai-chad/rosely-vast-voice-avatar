"""Rosely MuseTalk 1.5 clip POC. Runs upstream prepared-avatar inference with a subprocess.

This deliberately does not claim streaming/WebRTC support. No model is hot-held across
requests; startup and render latency must be benchmarked before building live calling.
"""
from __future__ import annotations
import asyncio
import hmac
import logging
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse

LOG = logging.getLogger("rosely.avatar")
BUNDLE_ROOT = Path(os.getenv("BUNDLE_ROOT", "/workspace/rosely-voice-musetalk-v1"))
MUSE_DIR = BUNDLE_ROOT / "source" / "MuseTalk"
MUSE_PY = Path(os.getenv("MUSE_PYTHON", "/workspace/rosely-venvs/muse/bin/python"))
MEDIA_ROOT = Path(os.getenv("AVATAR_MEDIA_ROOT", "/workspace/rosely-avatar-data"))
AVATAR_PORT = int(os.getenv("AVATAR_PORT", "8003"))
API_KEY = os.environ.get("VOICE_API_KEY", "")
ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MAX_VIDEO_BYTES = int(os.getenv("MAX_AVATAR_SOURCE_BYTES", str(25 * 1024 * 1024)))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AVATAR_AUDIO_BYTES", str(10 * 1024 * 1024)))
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AVATAR_AUDIO_SECONDS", "12"))
RENDER_TIMEOUT = int(os.getenv("AVATAR_INFERENCE_TIMEOUT", "900"))
BATCH_SIZE = int(os.getenv("MUSE_BATCH_SIZE", "4"))
model_lock = asyncio.Lock()
app = FastAPI(title="Rosely MuseTalk Clip Renderer", version="0.1.0")


def authorize(key: str | None) -> None:
    if not API_KEY or not key or not hmac.compare_digest(key, API_KEY):
        raise HTTPException(401, "invalid API key")


def checked_id(avatar_id: str) -> str:
    if not ID_RE.fullmatch(avatar_id):
        raise HTTPException(400, "avatar_id must be 1-64 letters, numbers, _ or -")
    return avatar_id


def paths(avatar_id: str):
    aid = checked_id(avatar_id)
    base = MUSE_DIR / "results" / "v15" / "avatars" / aid
    return base, MEDIA_ROOT / "sources" / aid / "source.mp4"


def run_cmd(argv: list[str], *, cwd: Path | None = None, timeout: int = 180):
    cp = subprocess.run(argv, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, timeout=timeout)
    if cp.returncode:
        LOG.error("command failed (%s): %s", cp.returncode, cp.stdout[-10000:])
        raise RuntimeError(f"process failed with code {cp.returncode}: {cp.stdout[-1800:]}")
    LOG.info("command completed: %s", argv[:5])
    return cp.stdout


def infer_config(avatar_id: str, source: Path, audio: Path | None = None, label: str = "") -> Path:
    """Create valid YAML for upstream realtime CLI, with sanitized paths/identifiers."""
    import json
    conf = MEDIA_ROOT / "configs" / f"{uuid.uuid4().hex}.yaml"
    conf.parent.mkdir(parents=True, exist_ok=True)
    # JSON string scalars are valid YAML; all paths generated internally.
    txt = (f"{avatar_id}:\n  preparation: {'False' if audio else 'True'}\n"
           f"  bbox_shift: 0\n  video_path: {json.dumps(str(source))}\n"
           f"  audio_clips:\n")
    if audio:
        txt += f"    {label}: {json.dumps(str(audio))}\n"
    else:
        txt += "    {}\n"  # fixed below to plain mapping
        txt = txt.replace("  audio_clips:\n    {}\n", "  audio_clips: {}\n")
    conf.write_text(txt)
    return conf


def run_muse(config: Path):
    return run_cmd([
        str(MUSE_PY), "-m", "scripts.realtime_inference",
        "--inference_config", str(config),
        "--unet_model_path", "models/musetalkV15/unet.pth",
        "--unet_config", "models/musetalkV15/musetalk.json",
        "--version", "v15", "--fps", "25", "--batch_size", str(BATCH_SIZE),
    ], cwd=MUSE_DIR, timeout=RENDER_TIMEOUT)


def probe_duration(path: Path) -> float:
    cp = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
                        capture_output=True, text=True, timeout=20)
    if cp.returncode:
        raise HTTPException(400, "invalid media file")
    try:
        value = float(cp.stdout.strip())
        if value <= 0 or value != value:
            raise ValueError("invalid duration")
        return value
    except ValueError:
        raise HTTPException(400, "media duration could not be determined")


@app.get("/health")
async def health():
    essential = [
        MUSE_DIR / "models/musetalkV15/unet.pth",
        MUSE_DIR / "models/musetalkV15/musetalk.json",
        MUSE_DIR / "models/whisper/pytorch_model.bin",
        MUSE_DIR / "models/sd-vae/diffusion_pytorch_model.bin",
    ]
    present = all(p.is_file() for p in essential)
    return {"ok": present, "service": "avatar", "mode": "clip-only",
            "weights_present": present, "busy": model_lock.locked(), "batch_size": BATCH_SIZE}


@app.post("/v1/avatar/prepare")
async def prepare(avatar_id: str = Form(...), file: UploadFile = File(...),
                  x_api_key: str | None = Header(default=None)):
    authorize(x_api_key)
    base, source = paths(avatar_id)
    if base.exists():
        raise HTTPException(409, "avatar already prepared; use a different avatar_id")
    content = await file.read(MAX_VIDEO_BYTES + 1)
    if not content or len(content) > MAX_VIDEO_BYTES:
        raise HTTPException(413, "empty or oversized source image/video")
    if model_lock.locked():
        raise HTTPException(429, "avatar renderer busy; retry")
    async with model_lock:
        # Recheck after lock, otherwise concurrent calls may collide.
        if base.exists():
            raise HTTPException(409, "avatar already prepared")
        source.parent.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".webm"}:
            raise HTTPException(400, "upload PNG/JPEG/WEBP image or MP4/MOV/WEBM video")
        image = suffix in {".png", ".jpg", ".jpeg", ".webp"}
        upload = MEDIA_ROOT / "tmp" / f"{uuid.uuid4().hex}{suffix}"
        upload.parent.mkdir(parents=True, exist_ok=True)
        upload.write_bytes(content)
        try:
            args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
            if image:
                args += ["-loop", "1"]
            args += ["-i", str(upload), "-t", "2", "-vf",
                     "fps=25,scale=720:720:force_original_aspect_ratio=decrease,"
                     "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                     "-an", "-pix_fmt", "yuv420p", str(source)]
            await asyncio.to_thread(run_cmd, args)
            if not source.exists() or not source.stat().st_size:
                raise RuntimeError("ffmpeg created an empty source video")
            conf = infer_config(avatar_id, source)
            try:
                await asyncio.to_thread(run_muse, conf)
            finally:
                conf.unlink(missing_ok=True)
            if not (base / "latents.pt").is_file():
                raise RuntimeError("MuseTalk did not generate cached avatar latents")
            return {"ok": True, "avatar_id": avatar_id, "prepared": True}
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            shutil.rmtree(base, ignore_errors=True)
            source.unlink(missing_ok=True)
            raise HTTPException(502, f"avatar preparation failed: {str(exc)[-800:]}")
        finally:
            upload.unlink(missing_ok=True)


@app.post("/v1/avatar/render")
async def render(background_tasks: BackgroundTasks, avatar_id: str = Form(...),
                 audio: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    authorize(x_api_key)
    base, source = paths(avatar_id)
    if not (base / "latents.pt").is_file() or not source.is_file():
        raise HTTPException(404, "avatar not prepared")
    content = await audio.read(MAX_AUDIO_BYTES + 1)
    if not content or len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "empty or oversized audio")
    if model_lock.locked():
        raise HTTPException(429, "avatar renderer busy; retry")
    async with model_lock:
        token = uuid.uuid4().hex
        audio_path = MEDIA_ROOT / "audio" / f"{token}.wav"
        raw = MEDIA_ROOT / "tmp" / f"{token}.bin"
        raw.parent.mkdir(parents=True, exist_ok=True)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(content)
        label = f"clip_{token}"
        try:
            await asyncio.to_thread(run_cmd, [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(raw), "-ac", "1", "-ar", "16000", str(audio_path),
            ])
            seconds = await asyncio.to_thread(probe_duration, audio_path)
            if seconds > MAX_AUDIO_SECONDS:
                raise HTTPException(413, f"audio longer than {MAX_AUDIO_SECONDS}s POC cap")
            conf = infer_config(avatar_id, source, audio_path, label)
            try:
                await asyncio.to_thread(run_muse, conf)
            finally:
                conf.unlink(missing_ok=True)
            video = base / "vid_output" / f"{label}.mp4"
            if not video.is_file() or video.stat().st_size == 0:
                raise RuntimeError("MuseTalk did not produce an MP4")
            background_tasks.add_task(video.unlink, missing_ok=True)
            return FileResponse(video, media_type="video/mp4", filename=f"{avatar_id}.mp4",
                                background=background_tasks)
        except HTTPException:
            raise
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(502, f"MuseTalk inference failed: {str(exc)[-800:]}")
        finally:
            raw.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
