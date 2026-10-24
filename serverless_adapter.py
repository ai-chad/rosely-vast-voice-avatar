"""Loopback-only JSON adapter for Vast PyWorker, authenticates to Rosely gateway.
Vast PyWorker forwards JSON bodies; multipart/MP4 routes cannot be proxied unchanged.
"""
import os
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request

UPSTREAM = 'http://127.0.0.1:' + os.getenv('GATEWAY_PORT', '8000')
SECRET = os.getenv('GATEWAY_API_KEY', '')
client = None

@asynccontextmanager
async def lifespan(app):
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=960, write=120, pool=10))
    try:
        yield
    finally:
        await client.aclose()
        client = None

app = FastAPI(title='Rosely Vast JSON Serverless Adapter', lifespan=lifespan)

@app.get('/health')
async def health():
    try:
        response = await client.get(UPSTREAM + '/health')
        data = response.json()
        if response.status_code != 200 or not data.get('ok'):
            raise HTTPException(503, detail=data)
        return {'ok': True, 'service': 'rosely-serverless-adapter', 'gateway': data}
    except httpx.HTTPError as exc:
        raise HTTPException(503, detail=str(exc)) from exc

@app.post('/v1/tts')
@app.post('/v1/stt')
@app.post('/v1/avatar/generate')
@app.post('/v1/ditto/warm')
async def proxy(request: Request):
    if not SECRET:
        raise HTTPException(503, 'missing internal gateway key')
    payload = await request.json()
    try:
        response = await client.post(UPSTREAM + request.url.path, json=payload,
                                     headers={'X-API-Key': SECRET})
    except httpx.HTTPError as exc:
        raise HTTPException(502, f'gateway unreachable: {exc}') from exc
    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:1000]
        raise HTTPException(response.status_code, detail=detail)
    return response.json()
