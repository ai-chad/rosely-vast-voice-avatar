"""Lightweight tests of request validation; GPU and S3 integration tested on Vast."""
import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def load_file(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_avatar_input_contract():
    avatar = load_file('avatar_server')
    avatar.API_KEY = 'secret-for-test'
    with TestClient(avatar.app) as client:
        assert client.get('/health').status_code == 200
        invalid = client.post('/v1/avatar/prepare', data={'avatar_id': 'bad/id'},
                              files={'file': ('demo.png', b'data')}, headers={'X-API-Key': 'secret-for-test'})
        assert invalid.status_code == 400
        unauth = client.post('/v1/avatar/prepare', data={'avatar_id': 'demo'},
                             files={'file': ('demo.png', b'data')})
        assert unauth.status_code == 401
        missing = client.post('/v1/avatar/render', data={'avatar_id': 'never_created'},
                              files={'audio': ('x.wav', b'audio')},
                              headers={'X-API-Key': 'secret-for-test'})
        assert missing.status_code == 404


def test_gateway_stt_tts_contract():
    gateway = load_file('voice_gateway')
    assert gateway.TTSRequest(input='hello', voice='en_female_happy').voice == 'en_female_happy'
    assert gateway.STTRequest(audio_base64='YWJj').filename == 'audio.wav'
    gateway.GATEWAY_API_KEY = 'secret-for-test'
    with TestClient(gateway.app) as client:
        assert client.post('/v1/tts', json={'input': 'Hi'}).status_code == 401
        assert client.post('/v1/stt', json={'audio_base64': '!!'},
                           headers={'X-API-Key': 'secret-for-test'}).status_code == 400
