"""No-GPU checks for the Ditto engine: acting controls and the gateway request contract."""
import asyncio

import pytest
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_file(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_every_llm_voice_style_has_an_emotion():
    acting = load_file('acting')
    llm_styles = {"neutral", "calm", "happy", "excited", "playful", "shy", "seductive", "affectionate",
                  "sad", "worried", "angry", "tired", "surprised", "confused", "frustrated"}
    assert set(acting.EMOTIONS) == llm_styles
    assert all(0 <= label < 8 for labels, _, _ in acting.EMOTIONS.values() for label in labels)


def test_controls_shape_gestures_and_fallbacks():
    acting = load_file('acting')
    labels, ctrl = acting.controls('happy', 'nod', 50)
    assert labels == [3]
    assert max(abs(f.get('delta_pitch', 0)) for f in ctrl.values()) > 3          # visible nod
    assert all(f['delta_exp'].shape == (1, 63) for f in ctrl.values() if 'delta_exp' in f)
    _, shake = acting.controls('neutral', 'shake', 50)
    assert any('delta_yaw' in f for f in shake.values()) and not any('delta_exp' in f for f in shake.values())
    labels, plain = acting.controls('ecstatic', 'backflip', 10)                  # unknown -> neutral, none
    assert labels == [4] and plain == {}
    _, wink = acting.controls('neutral', 'wink', 3)                              # very short clips stay in range
    assert set(wink) <= {0, 1, 2}


def test_gateway_requests_are_ditto_with_acting():
    gateway = load_file('voice_gateway')
    request = gateway.GenerateAvatarRequest(image_base64='eA==', input='hi', emotion='happy', gesture='nod')
    assert (request.engine, request.emotion, request.gesture) == ('ditto', 'happy', 'nod')
    assert gateway.GenerateAvatarRequest(image_base64='eA==', input='hi', engine='musetalk').engine == 'musetalk'  # old backends: ignored


def test_video_portraits_are_refused(monkeypatch):
    from fastapi import HTTPException
    gateway = load_file('voice_gateway')
    gateway.GATEWAY_API_KEY = 'k'
    gateway.client = object()
    request = gateway.GenerateAvatarRequest(image_base64='eA==', image_filename='me.mp4', input='hi')
    with pytest.raises(HTTPException) as error:
        asyncio.run(gateway.generate_avatar(request, x_api_key='k'))
    assert error.value.status_code == 400 and 'still portrait' in error.value.detail


def test_ditto_failure_is_an_error_not_a_fallback(monkeypatch):
    import httpx
    from fastapi import HTTPException
    gateway = load_file('voice_gateway')

    class Down:
        async def post(self, *args, **kwargs):
            raise httpx.ConnectError('down')
    gateway.client = Down()
    with pytest.raises(HTTPException) as error:
        asyncio.run(gateway.render_ditto(b'img', '.png', b'wav', 'happy', 'nod'))
    assert error.value.status_code == 502


def test_tts_does_not_wait_for_the_gpu_lock():
    gateway = load_file('voice_gateway')
    gateway.GATEWAY_API_KEY = 'k'

    class Client:
        async def post(self, url, **kwargs):
            return type('R', (), {'status_code': 200, 'content': b'RIFF', 'headers': {'content-type': 'audio/wav'}})()
    gateway.client = Client()

    async def run():
        async with gateway.gpu_request_lock:  # a clip is rendering
            return await asyncio.wait_for(gateway.tts(gateway.TTSRequest(input='hi'), x_api_key='k'), 1)
    assert asyncio.run(run())['size_bytes'] == 4


def test_renders_thin_to_20fps_by_skipping_every_fifth_frame(monkeypatch):
    import sys
    import types
    written = {}

    class Writer:
        def __init__(self, path, fps=25):
            written.update(path=path, fps=fps)
    fake = types.ModuleType('core.atomic_components.writer')
    fake.VideoWriterByImageIO = Writer
    monkeypatch.setitem(sys.modules, 'core.atomic_components.writer', fake)
    monkeypatch.setitem(sys.modules, 'core', types.ModuleType('core'))
    monkeypatch.setitem(sys.modules, 'core.atomic_components', types.ModuleType('core.atomic_components'))
    server = load_file('ditto_server')
    queued = []
    sdk = types.SimpleNamespace(writer=types.SimpleNamespace(close=lambda: None), tmp_output_path='/tmp/x.mp4',
                                warp_f3d_queue=types.SimpleNamespace(put=lambda item, *a, **k: queued.append(item)))
    server.thin_to_20fps(sdk)
    for frame in range(10):
        sdk.warp_f3d_queue.put(frame)
    sdk.warp_f3d_queue.put(None)
    assert written == {'path': '/tmp/x.mp4', 'fps': 20}
    assert queued == [0, 1, 2, 3, 5, 6, 7, 8, None]   # 4 of every 5 frames, and the end marker


def test_next_parts_speech_is_synthesized_after_this_parts_and_reused(monkeypatch):
    gateway = load_file('voice_gateway')
    gateway.GATEWAY_API_KEY = 'k'
    order = []

    async def synthesize(text, voice):
        order.append(text)
        await asyncio.sleep(0.01)
        return f'wav:{text}'.encode()

    async def render(image, suffix, audio, emotion, gesture, motion=None, offset=0):
        order.append(f'render:{audio.decode()}')
        await asyncio.sleep(0.05)   # the face renders while the next part's speech is synthesized
        return b'mp4', 'still'
    monkeypatch.setattr(gateway, 'synthesize', synthesize)
    monkeypatch.setattr(gateway, 'render_ditto', render)
    gateway.client = object()

    async def run():
        first = gateway.GenerateAvatarRequest(image_base64='eA==', input='First part,', voice='v',
                                              prefetch_inputs=['the rest.'])
        await gateway.generate_avatar(first, x_api_key='k')
        second = gateway.GenerateAvatarRequest(image_base64='eA==', input='the rest.', voice='v')
        await gateway.generate_avatar(second, x_api_key='k')
    asyncio.run(run())
    # Part 1's speech first, then part 2's alongside part 1's render, never synthesized twice.
    assert order == ['First part,', 'render:wav:First part,', 'the rest.', 'render:wav:the rest.']
    assert gateway.prefetched_speech == {}
