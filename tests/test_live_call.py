"""No-GPU checks for live calls: turn detection, frame/audio pairing, chunk padding, gateway route."""
import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_file(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


live_media = load_file('live_media')
FRAME = b'\x00\x00' * 480  # 30 ms at 16 kHz


def run_detector(pattern, **kw):
    """pattern: string of 'v' (voiced) / '.' (silent) 30 ms frames."""
    flags = iter(pattern)
    detector = live_media.TurnDetector(lambda pcm: next(flags) == 'v', **kw)
    return [(i, r if isinstance(r, str) else (r[0], len(r[1]) // len(FRAME)))
            for i, r in enumerate(detector.feed(FRAME) for _ in pattern) if r]


def test_turn_needs_sustained_speech_then_silence():
    events = run_detector('..' + 'v' * 10 + '.' * 30, start_ms=240, end_ms=600, preroll_ms=90)
    assert events[0] == (9, 'speech')  # 8 voiced frames (240 ms) in a row
    index, (kind, frames) = events[1]
    assert kind == 'utterance' and index == 31  # 20 silent frames (600 ms) later
    assert frames == 3 + 2 + 3  # preroll (incl. the voiced start) + rest of speech + 90 ms tail


def test_blips_are_not_turns():
    assert run_detector('vv..vvv....v.' * 4) == []


def test_long_speech_is_cut_at_the_cap():
    events = run_detector('v' * 20, start_ms=60, max_ms=600)
    assert [e[1] if isinstance(e[1], str) else e[1][0] for e in events] == ['speech', 'utterance']


def test_reply_pairs_each_frame_with_its_audio():
    timeline = live_media.ReplyTimeline(fps=20, start_buffer_frames=2)
    spf = live_media.OUT_RATE // 20
    timeline.add_audio(np.arange(spf * 3, dtype=np.float32))
    timeline.add_frame('f0')
    assert timeline.take() == 'wait'  # buffering before the first frame
    timeline.add_frame('f1')
    frame, audio = timeline.take()
    assert frame == 'f0' and audio[0] == 0 and len(audio) == spf
    frame, audio = timeline.take()
    assert frame == 'f1' and audio[0] == spf
    assert timeline.take() == 'wait'  # next frame still rendering: hold, do not skip audio
    timeline.audio_complete = timeline.render_done = True
    frame, audio = timeline.take()  # renderer stopped early: last face, remaining audio
    assert frame == 'f1' and audio[0] == 2 * spf
    assert timeline.take() == 'end'


def test_tail_frames_past_the_audio_are_dropped():
    timeline = live_media.ReplyTimeline(fps=20, start_buffer_frames=0)
    timeline.add_audio(np.zeros(live_media.OUT_RATE // 20, np.float32))
    for name in ('a', 'b', 'c'):
        timeline.add_frame(name)
    timeline.audio_complete = True
    assert timeline.take()[0] == 'a'
    assert timeline.take() == 'end'


def test_abort_ends_the_reply():
    timeline = live_media.ReplyTimeline(fps=20, start_buffer_frames=0)
    timeline.add_audio(np.zeros(live_media.OUT_RATE, np.float32))
    timeline.add_frame('a')
    timeline.abort()
    timeline.add_frame('b')
    assert timeline.take() == 'end' and not timeline.frames


def test_chunk_padding_keeps_both_timelines_whole():
    if importlib.util.find_spec('acting') is None:
        pytest.skip('acting.py not importable')
    live_ditto = load_file('live_ditto')
    for samples in (1, 4799, 4800, 12345):
        pad = live_ditto.pad_to_chunk(samples, live_media.OUT_RATE)
        total = samples + pad
        assert total % 4800 == 0  # 0.2 s at 24 kHz
        assert (total * 16000 // live_media.OUT_RATE) % live_ditto.CHUNK_SAMPLES == 0
    assert [live_ditto.kept(i) for i in range(10)] == [True, True, True, True, False] * 2


def test_gateway_refuses_live_join_when_disabled(monkeypatch):
    pytest.importorskip('fastapi')
    monkeypatch.setenv('GATEWAY_API_KEY', 'k')
    monkeypatch.setenv('LIVE_CALL_ENABLED', '0')
    gateway = load_file('voice_gateway')
    from fastapi.testclient import TestClient
    with TestClient(gateway.app) as client:
        response = client.post('/v1/live/join', json={}, headers={'X-API-Key': 'k'})
    assert response.status_code == 409 and response.json()['detail'] == {'error': 'live_call_unavailable'}


def test_quick_start_splits_a_long_opening_at_its_first_comma():
    assert live_media.quick_start(["Wait until you hear this, it is honestly the best news all week!", "Wow."]) == [
        "Wait until you hear this,", "it is honestly the best news all week!", "Wow."]
    too_short_a_head = ["Oh my gosh, you actually got the job after all those interviews?"]
    assert live_media.quick_start(too_short_a_head) == too_short_a_head
    short = ["Hey you, what's up?"]
    assert live_media.quick_start(short) == short
    no_comma = ["I really cannot believe that you actually did that today"]
    assert live_media.quick_start(no_comma) == no_comma


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv('DITTO_CLIP_DIR', str(tmp_path / 'clips'))
    module = load_file('live_agent')
    request = module.JoinRequest(
        call_id='00000000-0000-0000-0000-000000000001', livekit_url='wss://x.invalid', token='t' * 30,
        user_identity='user-1', turn_url='https://x.invalid/turn', call_secret='s' * 30, voice_id='en_female_happy',
        portrait_url='https://x.invalid/p.png',
        motion_clips=[{'clip_type': t, 'motion_sha256': c * 64, 'motion_url': 'https://s3.invalid/' + t}
                      for t, c in (('shy', 'c'), ('playful', 'b'), ('talk', 'a'))])
    session = module.CallSession(request)
    session.portrait_path = tmp_path / 'portrait.png'
    return module, session


def test_replies_pick_the_moods_clip_then_talk_then_the_portrait(agent):
    module, session = agent
    assert session.reply_source('playful') == (session.portrait_path, 'still')  # nothing registered yet
    session.clip_paths = {'talk': Path('talk.mp4'), 'playful': Path('playful.mp4')}
    assert session.reply_source('playful') == (Path('playful.mp4'), 'playful')
    assert session.reply_source('flirty') == (Path('talk.mp4'), 'talk')


def test_clips_register_talk_first_and_never_during_a_turn(agent, monkeypatch):
    module, session = agent
    for clip in session.request.motion_clips:  # already in the local store: no download
        path = module.ditto_sources.clip_path(clip.motion_sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'mp4')
    calls, warmed = [], []
    on_disk = {'a', 'b'}  # talk and playful are cached; shy must be registered by ditto_server

    class Registrations:
        last = 'disk'
        def cached(self, path, **kwargs):
            return 'disk' if Path(path).stem[:1] in on_disk else None
        def __call__(self, path, **kwargs):
            assert Path(path).stem[:1] in on_disk, 'the agent must never register on the GPU itself'
            calls.append((Path(path).stem[:1], session.in_turn))

    class Http:
        async def post(self, url, data, **kwargs):
            assert url.endswith('/v1/ditto/warm') and not session.in_turn
            warmed.append(data['motion_sha256'][:1])
            on_disk.add(data['motion_sha256'][:1])  # ditto_server saved its registration
            return type('R', (), {'raise_for_status': lambda self: None})()
    monkeypatch.setattr(module, 'registrations', Registrations())
    monkeypatch.setattr(module, 'registration_kwargs', lambda: {})
    monkeypatch.setattr(module, 'http', Http())

    async def run():
        session.in_turn = True
        task = asyncio.create_task(session.prepare_clips())
        await asyncio.sleep(0.5)
        # Cached clips load (CPU) even during a turn; the GPU warm-up waits for the turn to end.
        assert [c[0] for c in calls] == ['a', 'b'] and warmed == []
        session.in_turn = False
        await task
    asyncio.run(run())
    assert calls == [('a', True), ('b', True), ('c', False)]  # talk, playful, shy
    assert warmed == ['c']
    assert set(session.clip_paths) == {'talk', 'playful', 'shy'}


def test_registration_cache_reports_cached_clips_without_registering(tmp_path, monkeypatch):
    monkeypatch.setenv('DITTO_CLIP_DIR', str(tmp_path / 'clips'))
    sources = load_file('ditto_sources')
    cache = sources.RegistrationCache(lambda *a, **k: pytest.fail('must not register'), disk_dir=tmp_path / 'cache')
    clip = sources.clip_path('d' * 64)
    assert cache.cached(clip, max_dim=1920, crop_scale=2.3) is None
    (tmp_path / 'cache').mkdir()
    cache._disk_file(cache._key('d' * 64, 1920, {'crop_scale': 2.3})).write_bytes(b'x')
    assert cache.cached(clip, max_dim=1920, crop_scale=2.3) == 'disk'


def test_clip_sized_frames_play_and_fade_back_without_stopping_the_media(agent):
    module, session = agent
    session.video = True
    session.idle_rgb = np.zeros((96, 54, 3), np.uint8)  # portrait size
    pushed = []

    class Frame:
        def __init__(self, *args):
            self.args = args

    class AV:
        async def push(self, frame):
            pushed.append(frame.args)
            if len(pushed) > 60:
                session.stop_event.set()
            await asyncio.sleep(0)

        async def clear_queue(self):
            pass
    rtc = type('rtc', (), {'AudioFrame': Frame, 'VideoBufferType': type('T', (), {'RGB24': 4}),
                           'VideoFrame': lambda w, h, kind, data: Frame('video', w, h, len(data))})
    session.av = AV()
    reply = live_media.ReplyTimeline(fps=20, start_buffer_frames=0)
    reply.turn_started = 0.0
    reply.add_audio(np.zeros(live_media.OUT_RATE // 20 * 2, np.float32))
    for _ in range(2):
        reply.add_frame(np.full((80, 48, 3), 200, np.uint8))  # motion-clip size differs from the portrait
    reply.audio_complete = reply.render_done = True
    session.reply = reply

    async def no_state(*args, **kwargs):
        pass
    session.state = no_state
    asyncio.run(session.media_loop(rtc, None))
    videos = [p for p in pushed if p[0] == 'video']
    assert len(videos) > 10  # the reply, its fade back, then idle portrait frames keep flowing
    assert all(p[1:3] == (54, 96) and p[3] == 54 * 96 * 3 for p in videos)
