"""Unchanged Chatterbox Turbo voice API shape from rosely-vast-voice."""
import asyncio
import io
import os
import re
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import librosa
import torch
import torchaudio
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get('TTS_MODEL_PATH', '/workspace/rosely-voice-v1/models/chatterbox-turbo')
VOICES_DIR = Path(os.environ.get('VOICES_DIR', '/workspace/rosely-voice-v1/voices'))
CACHE_DIR = VOICES_DIR / '.conditionals'
API_KEY = os.environ.get('VOICE_API_KEY', '')
MAX_TEXT_CHARS = int(os.environ.get('TTS_MAX_TEXT_CHARS', '2500'))
MAX_VOICE_UPLOAD_BYTES = int(os.environ.get('TTS_MAX_VOICE_UPLOAD_BYTES', str(30 * 1024 * 1024)))
VOICE_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,80}$')
model = None
builtin_conds = None
infer_lock = asyncio.Lock()


def check_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail='invalid API key')


def validate_voice_id(voice_id: str):
    if not VOICE_ID_RE.fullmatch(voice_id):
        raise HTTPException(status_code=400, detail='voice id may contain only letters, numbers, _ and -')


def voice_path(voice_id: str) -> Path:
    validate_voice_id(voice_id)
    return VOICES_DIR / f'{voice_id}.wav'


def cache_path(voice_id: str) -> Path:
    validate_voice_id(voice_id)
    return CACHE_DIR / f'{voice_id}.pt'


def activate_voice(voice_id: str | None):
    global model
    if not voice_id:
        if builtin_conds is None:
            raise RuntimeError('built-in Chatterbox conditioning unavailable')
        model.conds = builtin_conds
        return
    wav = voice_path(voice_id)
    if not wav.exists():
        raise HTTPException(status_code=404, detail=f"voice '{voice_id}' not found")
    cached = cache_path(voice_id)
    if cached.exists() and cached.stat().st_mtime >= wav.stat().st_mtime:
        model.conds = Conditionals.load(cached, map_location='cpu').to(model.device)
        return
    model.prepare_conditionals(str(wav))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model.conds.save(cached)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, builtin_conds
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model = ChatterboxTurboTTS.from_local(MODEL_PATH, device='cuda')
    builtin_conds = model.conds
    yield
    model = None
    builtin_conds = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title='Rosely Chatterbox Turbo', lifespan=lifespan)


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1)
    voice: str | None = None
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)
    top_p: float = Field(default=0.95, ge=0.05, le=1.0)
    top_k: int = Field(default=1000, ge=1, le=2000)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)


@app.get('/health')
def health():
    return {
        'ok': model is not None, 'service': 'tts', 'model': 'chatterbox-turbo-350m',
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'allocated_mb': round(torch.cuda.memory_allocated(0) / 1024**2, 1) if torch.cuda.is_available() else 0,
        'reserved_mb': round(torch.cuda.memory_reserved(0) / 1024**2, 1) if torch.cuda.is_available() else 0,
    }


@app.get('/v1/voices')
def list_voices(x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    return {'voices': sorted(p.stem for p in VOICES_DIR.glob('*.wav'))}


@app.post('/v1/voices/{voice_id}')
async def upload_voice(voice_id: str, file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    validate_voice_id(voice_id)
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail='empty voice reference')
    if len(payload) > MAX_VOICE_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='voice reference too large')
    destination = voice_path(voice_id)
    src = None
    converted = None
    try:
        suffix = Path(file.filename or 'voice.wav').suffix or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(payload)
            src = Path(tmp.name)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_out:
            converted = Path(tmp_out.name)
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(src), '-ac', '1', '-ar', '24000', str(converted)], check=True)
        duration = float(librosa.get_duration(path=str(converted)))
        if duration <= 5.0:
            raise HTTPException(status_code=400, detail='reference audio must be longer than 5 seconds')
        converted.replace(destination)
        cache_path(voice_id).unlink(missing_ok=True)
        return {'ok': True, 'voice': voice_id, 'duration_seconds': round(duration, 2)}
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=400, detail='ffmpeg could not decode voice audio')
    finally:
        if src:
            src.unlink(missing_ok=True)
        if converted:
            converted.unlink(missing_ok=True)


@app.delete('/v1/voices/{voice_id}')
def delete_voice(voice_id: str, x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    validate_voice_id(voice_id)
    voice_path(voice_id).unlink(missing_ok=True)
    cache_path(voice_id).unlink(missing_ok=True)
    return {'ok': True}


@app.post('/v1/audio/speech')
async def speech(request: SpeechRequest, x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    text = request.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail='input is empty')
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=413, detail=f'input exceeds {MAX_TEXT_CHARS} characters; chunk long replies')
    async with infer_lock:
        await asyncio.to_thread(activate_voice, request.voice)
        wav = await asyncio.to_thread(model.generate, text, temperature=request.temperature, top_p=request.top_p,
                                       top_k=request.top_k, repetition_penalty=request.repetition_penalty)
    buf = io.BytesIO()
    torchaudio.save(buf, wav.cpu(), model.sr, format='wav')
    return Response(content=buf.getvalue(), media_type='audio/wav',
                    headers={'X-Sample-Rate': str(model.sr), 'Cache-Control': 'no-store'})
