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
