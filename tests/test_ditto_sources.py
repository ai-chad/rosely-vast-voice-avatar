"""No-GPU checks for motion-clip sources: clip store, registration cache, gateway intake."""
import asyncio
import base64
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_file(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.setenv('DITTO_CLIP_DIR', str(tmp_path / 'clips'))
    module = load_file('ditto_sources')
    # Persist synchronously so disk behaviour is deterministic in tests.
    monkeypatch.setattr(module.threading, 'Thread',
                        lambda target, args, daemon: type('T', (), {'start': lambda self: target(*args)})())
    return module


class FakeRegistrar:
    def __init__(self):
        self.calls = []

    def __call__(self, source_path, max_dim, n_frames, **crop):
        self.calls.append((source_path, n_frames))
        frames = 3
        return {'x_s_info_lst': [{'pitch': np.zeros(1, np.float32)} for _ in range(frames)],
                'f_s_lst': [np.full((1, 2, 2), i + 0.5, np.float32) for i in range(frames)],
                'img_rgb_lst': [np.zeros((4, 4, 3), np.uint8)] * frames,
                'M_c2o_lst': [np.eye(3, dtype=np.float32)] * frames, 'is_image_flag': False}


def clip(sources, data=b'motion clip bytes'):
    sha = hashlib.sha256(data).hexdigest()
    return sources.store_clip(data, sha), sha


def test_clip_store_checks_the_digest(sources):
    path, sha = clip(sources)
    assert path.read_bytes() == b'motion clip bytes' and path.name == f'{sha}.mp4'
    with pytest.raises(ValueError):
        sources.store_clip(b'other bytes', sha)
    with pytest.raises(ValueError):
        sources.clip_path('../../etc/passwd')


def test_registration_cache_tiers_and_isolation(sources, tmp_path):
    path, _ = clip(sources)
    registrar = FakeRegistrar()
    cache = sources.RegistrationCache(registrar, ram_items=2, disk_dir=tmp_path / 'cache', disk_bytes=1e9)
    first = cache(str(path), max_dim=1920, n_frames=50, crop_scale=2.3)
    assert cache.last == 'registered' and registrar.calls == [(str(path), sources.ALL_FRAMES)]
    first['x_s_info_lst'] = 'smoothed by Ditto setup'           # Ditto replaces this key in place
    first_again = cache(str(path), max_dim=1920, n_frames=200, crop_scale=2.3)
    assert cache.last == 'ram' and len(registrar.calls) == 1   # any reply length reuses the entry
    assert isinstance(first_again['x_s_info_lst'], list)
    # A new process (empty RAM) restores from disk, with features back in float32.
    restarted = sources.RegistrationCache(registrar, ram_items=0, disk_dir=tmp_path / 'cache', disk_bytes=1e9)
    restored = restarted(str(path), max_dim=1920, n_frames=50, crop_scale=2.3)
    assert restarted.last == 'disk' and len(registrar.calls) == 1
    assert restored['f_s_lst'][2].dtype == np.float32 and np.allclose(restored['f_s_lst'][2], 2.5)
    assert restarted.stats()['disk_items'] == 1


def test_still_images_and_foreign_paths_are_not_cached(sources, tmp_path):
    registrar = FakeRegistrar()
    cache = sources.RegistrationCache(registrar, disk_dir=tmp_path / 'cache')
    cache(str(tmp_path / 'character.png'), max_dim=1920, n_frames=40)
    cache(str(tmp_path / 'character.png'), max_dim=1920, n_frames=40)
    assert cache.last == 'uncached' and registrar.calls[-1] == (str(tmp_path / 'character.png'), 40)
    assert len(registrar.calls) == 2


def test_disk_cache_evicts_oldest_beyond_budget(sources, tmp_path):
    registrar = FakeRegistrar()
    cache = sources.RegistrationCache(registrar, ram_items=0, disk_dir=tmp_path / 'cache', disk_bytes=1)
    for i in range(3):
        path, _ = clip(sources, f'clip {i}'.encode())
        cache(str(path), max_dim=1920, n_frames=10)
    assert cache.stats()['disk_items'] <= 1


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setenv('DITTO_CLIP_DIR', str(tmp_path / 'clips'))
    monkeypatch.setenv('DITTO_ENABLED', '1')
    load_file('ditto_sources')
    return load_file('voice_gateway')


def request(gateway, **fields):
    return gateway.GenerateAvatarRequest(image_base64='eA==', input='hi', engine='ditto', **fields)


def test_gateway_stores_inline_clips_and_reuses_them(gateway):
    data = b'motion clip bytes'
    sha = hashlib.sha256(data).hexdigest()
    stored = asyncio.run(gateway.ensure_motion_clip(request(gateway, motion_sha256=sha,
                                                            motion_base64=base64.b64encode(data).decode())))
    assert stored == sha
    # Later turns only need the digest.
    assert asyncio.run(gateway.ensure_motion_clip(request(gateway, motion_sha256=sha))) == sha


def test_gateway_falls_back_to_the_still_portrait(gateway):
    data = b'motion clip bytes'
    wrong = hashlib.sha256(b'something else').hexdigest()
    assert asyncio.run(gateway.ensure_motion_clip(request(
        gateway, motion_sha256=wrong, motion_base64=base64.b64encode(data).decode()))) is None   # digest mismatch
    assert asyncio.run(gateway.ensure_motion_clip(request(
        gateway, motion_sha256=wrong, motion_url='http://insecure.example/clip.mp4'))) is None   # https only
    assert asyncio.run(gateway.ensure_motion_clip(request(gateway))) is None                    # no clip asked


def test_gateway_rejects_malformed_digests(gateway):
    with pytest.raises(Exception):
        request(gateway, motion_sha256='not-a-digest')
