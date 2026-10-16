"""Unchanged Qwen3-ASR HTTP contract from rosely-vast-voice."""
import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from qwen_asr import Qwen3ASRModel

MODEL_PATH = os.environ.get('ASR_MODEL_PATH', '/workspace/rosely-voice-musetalk-v1/models/Qwen3-ASR-0.6B')
API_KEY = os.environ.get('VOICE_API_KEY', '')
MAX_UPLOAD_BYTES = int(os.environ.get('ASR_MAX_UPLOAD_BYTES', str(50 * 1024 * 1024)))
model = None
infer_lock = asyncio.Lock()


def check_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail='invalid API key')


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    model = Qwen3ASRModel.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map='cuda:0',
        attn_implementation=os.environ.get('ASR_ATTN_IMPLEMENTATION', 'sdpa'),
        max_inference_batch_size=1,
        max_new_tokens=int(os.environ.get('ASR_MAX_NEW_TOKENS', '256')),
    )
    yield
    model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title='Rosely Qwen3 ASR', lifespan=lifespan)


@app.get('/health')
def health():
    return {
        'ok': model is not None, 'service': 'asr', 'model': 'Qwen3-ASR-0.6B',
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'allocated_mb': round(torch.cuda.memory_allocated(0) / 1024**2, 1) if torch.cuda.is_available() else 0,
        'reserved_mb': round(torch.cuda.memory_reserved(0) / 1024**2, 1) if torch.cuda.is_available() else 0,
    }


@app.post('/v1/audio/transcriptions')
async def transcribe(file: UploadFile = File(...), language: str | None = Form(default=None), x_api_key: str | None = Header(default=None)):
    check_key(x_api_key)
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail='empty audio file')
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail='audio file too large')
    suffix = Path(file.filename or 'audio.wav').suffix or '.wav'
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name
        async with infer_lock:
            results = await asyncio.to_thread(model.transcribe, audio=tmp_path, language=language or None)
        if not results:
            return {'text': '', 'language': language}
        result = results[0]
        return {'text': result.text, 'language': result.language}
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
