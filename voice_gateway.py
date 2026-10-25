"""Authenticated entrypoint for the Vast RTX 3090 worker: ASR, TTS, Ditto video clips, live calls."""
from __future__ import annotations
import asyncio
import base64
import binascii
import hmac
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

import ditto_sources

ASR_URL = f"http://127.0.0.1:{os.getenv('ASR_PORT', '8001')}"
TTS_URL = f"http://127.0.0.1:{os.getenv('TTS_PORT', '8002')}"
DITTO_URL = f"http://127.0.0.1:{os.getenv('DITTO_PORT', '8005')}"
# Live calls are opt-in per worker (LIVE_CALL_ENABLED=1): live_agent.py joins the call room.
LIVE_CALL_ENABLED = os.getenv('LIVE_CALL_ENABLED', '0') == '1'
LIVE_URL = f"http://127.0.0.1:{os.getenv('LIVE_PORT', '8006')}"
MAX_MOTION_CLIP_BYTES = int(os.getenv('MAX_MOTION_CLIP_BYTES', str(30 * 1024 * 1024)))
# Stand-in portrait for motion-clip renders (Ditto uses the clip, never this image).
PLACEHOLDER_PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')
IMAGE_REQUIRED = {'error': 'image_required'}
VOICE_API_KEY = os.getenv('VOICE_API_KEY', '')
GATEWAY_API_KEY = os.getenv('GATEWAY_API_KEY', VOICE_API_KEY)
MAX_STT_AUDIO_BYTES = int(os.getenv('MAX_STT_AUDIO_BYTES', str(50*1024*1024)))
MAX_AVATAR_FILE_BYTES = int(os.getenv('MAX_AVATAR_SOURCE_BYTES', str(25*1024*1024)))
MAX_AVATAR_AUDIO_BYTES = int(os.getenv('MAX_AVATAR_AUDIO_BYTES', str(10*1024*1024)))
client: httpx.AsyncClient | None = None
# Serializes Ditto clip renders and transcription. TTS is not under it: each model server has its
# own lock, and synthesizing a reply's next part while the current part renders shortens replies.
gpu_request_lock = asyncio.Lock()


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
    upstreams = {'asr': ASR_URL, 'tts': TTS_URL, 'ditto': DITTO_URL, **({'live': LIVE_URL} if LIVE_CALL_ENABLED else {})}
    statuses = await asyncio.gather(*(client.get(url + '/health') for url in upstreams.values()),
                                    return_exceptions=True)
    services = {}
    for name, result in zip(upstreams, statuses):
        if isinstance(result, BaseException):
            services[name] = {'ok': False, 'error': type(result).__name__}
        elif result.status_code != 200:
            services[name] = {'ok': False, 'http_status': result.status_code}
        else:
            services[name] = result.json()
    # The live agent is optional: turn-based voice/video stay healthy without it.
    healthy = all(s.get('ok') for name, s in services.items() if name != 'live')
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
        async with gpu_request_lock:
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


class GenerateAvatarRequest(BaseModel):
    """One spoken Ditto clip per request (Serverless: no state is shared between workers)."""
    # May be empty when a motion clip is sent: Ditto then renders from the clip alone.
    image_base64: str = ""
    audio_base64: str | None = None
    input: str | None = Field(default=None, max_length=2500)
    voice: str | None = None
    image_filename: str = "character.png"
    # Ditto is the only engine; older backends still send this field.
    engine: str = Field(default='ditto', max_length=16)
    # Per-reply acting controls (see acting.py).
    emotion: str = Field(default='neutral', max_length=32)
    gesture: str = Field(default='none', max_length=16)
    # Optional pre-generated motion clip for Ditto (body/hands/head), identified by its sha256.
    # Sent once per worker as an HTTPS URL or inline base64; later turns can send only the digest.
    motion_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    motion_url: str | None = Field(default=None, max_length=4096)
    motion_base64: str | None = None
    # Start point in the clip's back-and-forth walk (random per reply; continued per sentence).
    motion_offset_frames: int = Field(default=0, ge=0, le=1_000_000)
    # The reply's next parts: their speech is synthesized while this part's face renders, and
    # the next clip request for the same text and voice on this worker uses it (no TTS wait).
    prefetch_inputs: list[str] = Field(default_factory=list, max_length=4)


# Pre-synthesized speech for upcoming reply parts: (text, voice) -> task returning WAV bytes.
PREFETCH_LIMIT = 16
prefetched_speech: dict[tuple[str, str | None], asyncio.Task] = {}


async def synthesize(text: str, voice: str | None) -> bytes:
    assert client is not None
    try:
        speech = await client.post(TTS_URL + '/v1/audio/speech', headers=internal_headers(),
                                   json={'input': text, 'voice': voice})
    except httpx.HTTPError as exc:
        raise HTTPException(502, f'TTS request failed: {exc}') from exc
    if speech.status_code >= 400:
        raise upstream_error('tts', speech)
    return speech.content


def prefetch_speech(texts: list[str], voice: str | None) -> None:
    for text in texts:
        key = (text.strip(), voice)
        if key[0] and key not in prefetched_speech:
            task = asyncio.create_task(synthesize(key[0], voice))
            task.add_done_callback(lambda t: t.cancelled() or t.exception())  # never "unretrieved"
            prefetched_speech[key] = task
    while len(prefetched_speech) > PREFETCH_LIMIT:  # unclaimed (e.g. the next part went to another worker)
        prefetched_speech.pop(next(iter(prefetched_speech))).cancel()


async def speech_for(text: str, voice: str | None) -> bytes:
    """This text's speech: pre-synthesized by an earlier part's request if possible."""
    task = prefetched_speech.pop((text.strip(), voice), None)
    if task is not None:
        try:
            return await task
        except Exception:
            pass  # synthesize it now instead
    return await synthesize(text, voice)


async def ensure_motion_clip(request: GenerateAvatarRequest) -> str | None:
    """Digest of the motion clip once it is in the local store, or None (render the still portrait)."""
    return await ensure_clip(request.motion_sha256, request.motion_url, request.motion_base64)


async def ensure_clip(sha: str | None, url: str | None = None, inline: str | None = None) -> str | None:
    if not sha:
        return None
    if ditto_sources.clip_path(sha).exists():
        return sha
    try:
        if inline:
            if len(inline) > ((MAX_MOTION_CLIP_BYTES + 2) // 3) * 4 + 16:
                raise ValueError('motion clip too large')
            data = base64.b64decode(inline, validate=True)
        elif url and url.startswith('https://'):
            assert client is not None
            async with client.stream('GET', url, timeout=60) as response:
                response.raise_for_status()
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > MAX_MOTION_CLIP_BYTES:
                        raise ValueError('motion clip too large')
                    chunks.append(chunk)
            data = b''.join(chunks)
        else:
            return None
        await asyncio.to_thread(ditto_sources.store_clip, data, sha)
        return sha
    except Exception as exc:
        print(f'[gateway] motion clip unavailable, using the still portrait: {type(exc).__name__}', flush=True)
        return None


async def render_ditto(image: bytes, suffix: str, audio: bytes, emotion: str, gesture: str,
                       motion_sha256: str | None = None, motion_offset: int = 0) -> tuple[bytes, str]:
    """(Ditto clip, source used: "motion" or "still")."""
    assert client is not None
    try:
        rendered = await client.post(DITTO_URL + '/v1/ditto/render', headers=internal_headers(),
            data={'emotion': emotion, 'gesture': gesture, 'motion_sha256': motion_sha256 or '',
                  'motion_offset_frames': str(motion_offset)},
            files={'image': ('character' + suffix, image, 'application/octet-stream'),
                   'audio': ('speech.wav', audio, 'audio/wav')})
    except httpx.HTTPError as exc:
        raise HTTPException(502, f'Ditto unreachable: {type(exc).__name__}') from exc
    if rendered.status_code >= 400:
        raise upstream_error('ditto', rendered)
    return rendered.content, rendered.headers.get('X-Ditto-Source', 'still')


class WarmClip(BaseModel):
    motion_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    motion_url: str | None = Field(default=None, max_length=4096)


class WarmRequest(BaseModel):
    clips: list[WarmClip] = Field(default_factory=list, max_length=8)


@app.post('/v1/ditto/warm')
async def warm_clips(request: WarmRequest, x_api_key: str | None = Header(default=None)):
    """Download and pre-register a character's motion clips before its next replies need them.

    This does not take the gateway's GPU lock: transcription and speech must never wait
    behind a warm-up (a cold batch takes ~10-20 s, right when a video chat opens). Ditto
    serializes its own work, registering one clip at a time, so a reply render waits at
    most for the clip being registered.
    """
    authorize(x_api_key)
    assert client is not None
    results = {}
    for clip in request.clips:
        sha = await ensure_clip(clip.motion_sha256, clip.motion_url)
        if sha is None:
            results[clip.motion_sha256[:12]] = 'unavailable'
            continue
        try:
            warmed = await client.post(DITTO_URL + '/v1/ditto/warm', headers=internal_headers(),
                                       data={'motion_sha256': sha})
            results[sha[:12]] = warmed.json().get('cache') if warmed.status_code == 200 else f'http {warmed.status_code}'
        except httpx.HTTPError as exc:
            results[sha[:12]] = f'error {type(exc).__name__}'
    return {'ok': True, 'clips': results}


@app.post('/v1/live/join')
async def live_join(request: Request, x_api_key: str | None = Header(default=None)):
    """Ask this worker's live agent to join a call room as the character (see live_agent.py)."""
    authorize(x_api_key)
    if not LIVE_CALL_ENABLED:
        raise HTTPException(409, {'error': 'live_call_unavailable'})
    assert client is not None
    try:
        joined = await client.post(LIVE_URL + '/v1/live/join', headers=internal_headers(),
                                   json=await request.json(), timeout=60)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f'live agent unreachable: {type(exc).__name__}') from exc
    if joined.status_code >= 400:
        raise upstream_error('live', joined)
    return joined.json()


@app.post('/v1/avatar/generate')
async def generate_avatar(request: GenerateAvatarRequest, x_api_key: str | None = Header(default=None)):
    authorize(x_api_key)
    assert client is not None
    if bool(request.audio_base64) == bool(request.input):
        raise HTTPException(400, 'provide exactly one of audio_base64 or input')
    motion_ready = None
    if not request.image_base64:
        # Clip-only request (saves uploading the portrait each sentence). Without a usable
        # clip, ask the caller to resend with the portrait instead of guessing.
        motion_ready = await ensure_motion_clip(request)
        if motion_ready is None:
            raise HTTPException(409, IMAGE_REQUIRED)
        image, suffix = PLACEHOLDER_PNG, '.png'
    else:
        # Reject oversized b64 before materializing decoded files.
        if len(request.image_base64) > ((MAX_AVATAR_FILE_BYTES + 2) // 3) * 4 + 16:
            raise HTTPException(413, 'image too large')
        try:
            image = base64.b64decode(request.image_base64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(400, 'invalid image_base64')
        if not image or len(image) > MAX_AVATAR_FILE_BYTES:
            raise HTTPException(413, 'empty or oversized image')
        suffix = '.' + request.image_filename.rsplit('.', 1)[-1].lower() if '.' in request.image_filename else ''
        if suffix not in {'.png', '.jpg', '.jpeg', '.webp'}:
            raise HTTPException(400, 'Ditto needs a still portrait (png, jpg or webp)')
    async with gpu_request_lock:
        if request.audio_base64 is not None:
            if len(request.audio_base64) > ((MAX_AVATAR_AUDIO_BYTES + 2) // 3) * 4 + 16:
                raise HTTPException(413, 'audio too large')
            try:
                audio = base64.b64decode(request.audio_base64, validate=True)
            except (binascii.Error, ValueError):
                raise HTTPException(400, 'invalid audio_base64')
            if not audio or len(audio) > MAX_AVATAR_AUDIO_BYTES:
                raise HTTPException(413, 'empty or oversized audio')
        else:
            audio = await speech_for(request.input, request.voice)
        # This part's speech is done: the TTS server is free to work on the next parts while
        # Ditto renders this one.
        prefetch_speech(request.prefetch_inputs, request.voice)
        motion = motion_ready or await ensure_motion_clip(request)
        video, source = await render_ditto(image, suffix, audio, request.emotion, request.gesture, motion,
                                           request.motion_offset_frames)
    # Short clips travel as base64 in the JSON response (the backend stores them in S3).
    return {'video_base64': base64.b64encode(video).decode('ascii'), 'content_type': 'video/mp4',
            'size_bytes': len(video), 'engine': 'ditto', 'ditto_source': source}
