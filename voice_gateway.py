"""Authenticated entrypoint for dedicated Vast RTX 3090. Voice JSON routes retained."""
from __future__ import annotations
import asyncio
import base64
import binascii
import hmac
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

ASR_URL = f"http://127.0.0.1:{os.getenv('ASR_PORT', '8001')}"
TTS_URL = f"http://127.0.0.1:{os.getenv('TTS_PORT', '8002')}"
AVATAR_URL = f"http://127.0.0.1:{os.getenv('AVATAR_PORT', '8003')}"
VOICE_API_KEY = os.getenv('VOICE_API_KEY', '')
GATEWAY_API_KEY = os.getenv('GATEWAY_API_KEY', VOICE_API_KEY)
MAX_STT_AUDIO_BYTES = int(os.getenv('MAX_STT_AUDIO_BYTES', str(50*1024*1024)))
MAX_AVATAR_FILE_BYTES = int(os.getenv('MAX_AVATAR_SOURCE_BYTES', str(25*1024*1024)))
MAX_AVATAR_AUDIO_BYTES = int(os.getenv('MAX_AVATAR_AUDIO_BYTES', str(10*1024*1024)))
client: httpx.AsyncClient | None = None


def authorize(key: str | None):
    if not GATEWAY_API_KEY or not key or not hmac.compare_digest(key, GATEWAY_API_KEY):
        raise HTTPException(401, "invalid API key")


def internal_headers():
    return {'X-API-Key': VOICE_API_KEY}


def upstream_error(service: str, response: httpx.Response):
    try:
        detail = response.json()
    except ValueError:
        detail = response.text[:1000]
    return HTTPException(response.status_code, {'service': service, 'upstream': detail})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=960, write=120, pool=10))
    try:
        yield
    finally:
        await client.aclose()
        client = None


app = FastAPI(title='Rosely Vast Voice + Avatar Gateway', version='1.0.0', lifespan=lifespan)


class STTRequest(BaseModel):
    audio_base64: str = Field(min_length=1)
    filename: str = 'audio.wav'
    content_type: str = 'audio/wav'
    language: str | None = None


class TTSRequest(BaseModel):
    input: str = Field(min_length=1, max_length=2500)
    voice: str | None = None
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)
    top_p: float = Field(default=0.95, ge=0.05, le=1.0)
    top_k: int = Field(default=1000, ge=1, le=2000)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)


@app.get('/health')
async def health():
    assert client is not None
    statuses = await asyncio.gather(*(client.get(url + '/health') for url in (ASR_URL, TTS_URL, AVATAR_URL)),
                                    return_exceptions=True)
    services = {}
    for name, result in zip(('asr', 'tts', 'avatar'), statuses):
        if isinstance(result, BaseException):
            services[name] = {'ok': False, 'error': type(result).__name__}
        elif result.status_code != 200:
            services[name] = {'ok': False, 'http_status': result.status_code}
        else:
            services[name] = result.json()
    healthy = all(s.get('ok') for s in services.values())
    return {'ok': healthy, 'service': 'voice-avatar-gateway', **services}


@app.post('/v1/stt')
async def stt(request: STTRequest, x_api_key: str | None = Header(default=None)):
    authorize(x_api_key)
    assert client is not None
    if len(request.audio_base64) > ((MAX_STT_AUDIO_BYTES + 2) // 3) * 4 + 16:
        raise HTTPException(413, 'audio is too large')
    try:
        audio = base64.b64decode(request.audio_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, 'invalid audio_base64')
    if not audio or len(audio) > MAX_STT_AUDIO_BYTES:
        raise HTTPException(413, 'empty or oversized audio')
    form_data = {'language': request.language} if request.language else {}
    try:
        result = await client.post(ASR_URL + '/v1/audio/transcriptions', headers=internal_headers(),
                                   files={'file': (request.filename, audio, request.content_type)}, data=form_data)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f'ASR error: {exc}')
    if result.status_code >= 400:
        raise upstream_error('asr', result)
    return result.json()


@app.post('/v1/tts')
async def tts(request: TTSRequest, x_api_key: str | None = Header(default=None)):
    authorize(x_api_key)
    assert client is not None
    try:
        result = await client.post(TTS_URL + '/v1/audio/speech', headers=internal_headers(),
                                   json=request.model_dump(exclude_none=True))
    except httpx.HTTPError as exc:
        raise HTTPException(502, f'TTS error: {exc}')
    if result.status_code >= 400:
        raise upstream_error('tts', result)
    return {'audio_base64': base64.b64encode(result.content).decode('ascii'),
            'content_type': result.headers.get('content-type', 'audio/wav'),
            'sample_rate': result.headers.get('x-sample-rate'), 'size_bytes': len(result.content)}


@app.post('/v1/avatar/prepare')
async def avatar_prepare(avatar_id: str = Form(...), file: UploadFile = File(...),
                         x_api_key: str | None = Header(default=None)):
    authorize(x_api_key)
    assert client is not None
    payload = await file.read(MAX_AVATAR_FILE_BYTES + 1)
    if not payload or len(payload) > MAX_AVATAR_FILE_BYTES:
        raise HTTPException(413, 'empty or oversized avatar input')
    try:
        result = await client.post(AVATAR_URL + '/v1/avatar/prepare', headers=internal_headers(),
                                   data={'avatar_id': avatar_id},
                                   files={'file': (file.filename or 'avatar.png', payload, file.content_type or 'application/octet-stream')})
    except httpx.HTTPError as exc:
        raise HTTPException(502, f'avatar prepare error: {exc}')
    if result.status_code >= 400:
        raise upstream_error('avatar', result)
    return result.json()


@app.post('/v1/avatar/render')
async def avatar_render(avatar_id: str = Form(...), audio: UploadFile = File(...),
                        x_api_key: str | None = Header(default=None)):
    authorize(x_api_key)
    assert client is not None
    payload = await audio.read(MAX_AVATAR_AUDIO_BYTES + 1)
    if not payload or len(payload) > MAX_AVATAR_AUDIO_BYTES:
        raise HTTPException(413, 'empty or oversized audio input')
    try:
        result = await client.post(AVATAR_URL + '/v1/avatar/render', headers=internal_headers(),
                                   data={'avatar_id': avatar_id},
                                   files={'audio': (audio.filename or 'speech.wav', payload, audio.content_type or 'audio/wav')})
    except httpx.HTTPError as exc:
        raise HTTPException(502, f'avatar render error: {exc}')
    if result.status_code >= 400:
        raise upstream_error('avatar', result)
    return Response(result.content, media_type='video/mp4',
                    headers={'Cache-Control': 'no-store', 'Content-Disposition': 'attachment; filename="avatar.mp4"'})
