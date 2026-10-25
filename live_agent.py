"""Rosely live call agent: joins a LiveKit room as the character (opt-in, LIVE_CALL_ENABLED=1).

The backend mints the room tokens and asks this worker to join (gateway route
/v1/live/join). Per call, on this GPU:
  user microphone -> voice activity turns -> local ASR -> backend turn (character
  LLM, policy, history, per-reply billing) -> local TTS per sentence -> Ditto
  online stream (video) -> LiveKit audio + video tracks, kept in sync.
Between replies the call shows the still portrait. Speaking over the character
stops it (barge-in). One call at a time per worker: it owns the streaming renderer.
Nothing here holds LiveKit or backend credentials beyond the per-call token and secret.
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import live_media
from live_media import OUT_RATE, ReplyTimeline, TurnDetector

LOG = logging.getLogger("uvicorn.error")
API_KEY = os.environ.get("VOICE_API_KEY", "")
ASR_URL = f"http://127.0.0.1:{os.getenv('ASR_PORT', '8001')}"
TTS_URL = f"http://127.0.0.1:{os.getenv('TTS_PORT', '8002')}"
BUNDLE_ROOT = Path(os.getenv("BUNDLE_ROOT", "/workspace/rosely-voice-musetalk-v1"))
DITTO_ROOT = Path(os.getenv("DITTO_ROOT", str(BUNDLE_ROOT / "source" / "ditto-talkinghead")))
DITTO_MODELS = Path(os.getenv("DITTO_MODELS", str(BUNDLE_ROOT / "models" / "ditto")))
WORK_DIR = Path(os.getenv("AVATAR_MEDIA_ROOT", "/workspace/rosely-avatar-data")) / "live"
MAX_PORTRAIT_BYTES = 25 * 1024 * 1024
PORTRAIT_LONG_SIDE = int(os.getenv("LIVE_PORTRAIT_LONG_SIDE", "960"))
START_BUFFER_S = float(os.getenv("LIVE_START_BUFFER_S", "0.15"))  # Ditto renders ~22 fps vs 20 played
JOIN_TIMEOUT_S = 60  # the caller must appear in the room within this
IDLE_TIMEOUT_S = int(os.getenv("LIVE_IDLE_TIMEOUT_S", "600"))
BARGE_IN_MS = 420  # speech needed to interrupt the character (echo-safe margin)
VOICE_ONLY = object()  # stands in for a video frame in voice calls

sdk = None
active: "CallSession | None" = None
http: httpx.AsyncClient | None = None


def authorize(key: str | None) -> None:
    if not API_KEY or not key or not hmac.compare_digest(key, API_KEY):
        raise HTTPException(401, "invalid API key")


def load_sdk():
    import sys
    sys.path.insert(0, str(DITTO_ROOT))
    os.chdir(DITTO_ROOT)
    from stream_pipeline_online import StreamSDK
    return StreamSDK(str(DITTO_MODELS / "ditto_cfg" / "v0.4_hubert_cfg_trt_online.pkl"),
                     str(DITTO_MODELS / "ditto_trt_Ampere_Plus"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sdk, http
    http = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=60, write=30, pool=10))
    started = time.monotonic()
    sdk = await asyncio.to_thread(load_sdk)
    LOG.info("Live agent: Ditto online models loaded in %.1fs", time.monotonic() - started)
    try:
        yield
    finally:
        if active:
            active.stop("shutdown")
        await http.aclose()


app = FastAPI(title="Rosely Live Call Agent", version="0.1.0", lifespan=lifespan)


class JoinRequest(BaseModel):
    call_id: str = Field(pattern=r"^[0-9a-f-]{36}$")
    mode: Literal["voice", "video"] = "video"
    livekit_url: str = Field(pattern=r"^wss?://")
    token: str = Field(min_length=20, max_length=4000)
    user_identity: str = Field(min_length=1, max_length=200)
    turn_url: str = Field(pattern=r"^https?://")
    call_secret: str = Field(min_length=20, max_length=200)
    portrait_url: str | None = None
    voice_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    language: str | None = None
    max_seconds: int = Field(default=1800, ge=60, le=7200)


@app.get("/health")
async def health():
    return {"ok": sdk is not None, "service": "live-agent", "busy": bool(active and not active.finished)}


@app.post("/v1/live/join")
async def join(request: JoinRequest, x_api_key: str | None = Header(default=None)):
    global active
    authorize(x_api_key)
    if sdk is None:
        raise HTTPException(503, "live agent is starting")
    if active and not active.finished:
        if active.request.call_id == request.call_id:
            return {"ok": True, "call_id": request.call_id, "joined": "already"}
        raise HTTPException(409, {"error": "live_call_busy"})
    session = CallSession(request)
    await session.prepare()  # portrait fetched and checked before we say yes
    active = session
    session.task = asyncio.create_task(session.run())
    return {"ok": True, "call_id": request.call_id}


class CallSession:
    def __init__(self, request: JoinRequest):
        self.request = request
        self.video = request.mode == "video"
        self.finished = False
        self.task: asyncio.Task | None = None
        self.stop_event = asyncio.Event()
        self.end_reason = ""
        self.reply: ReplyTimeline | None = None
        self.stream = None  # live_ditto.DittoStream of the current reply
        self.speaking = False
        self.seq = 0
        self.last_activity = time.monotonic()
        self.utterances: asyncio.Queue = asyncio.Queue()
        self.renderer_idle = asyncio.Event()  # the streaming renderer is free for the next reply
        self.renderer_idle.set()
        self.portrait_path: Path | None = None
        self.idle_rgb: np.ndarray | None = None
        self.room = None
        self.av = None

    # -- setup ---------------------------------------------------------------------
    async def prepare(self):
        if not self.video:
            return
        if not self.request.portrait_url:
            raise HTTPException(422, "video calls need a portrait_url")
        try:
            response = await http.get(self.request.portrait_url)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"portrait download failed: {type(exc).__name__}") from exc
        if response.status_code != 200 or not response.content or len(response.content) > MAX_PORTRAIT_BYTES:
            raise HTTPException(422, "portrait unavailable")
        self.portrait_path, self.idle_rgb = await asyncio.to_thread(self._portrait, response.content)

    def _portrait(self, data: bytes):
        from PIL import Image
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB")
            image.thumbnail((PORTRAIT_LONG_SIDE, PORTRAIT_LONG_SIDE))
            w, h = image.size
            image = image.crop((0, 0, w - w % 2, h - h % 2))  # I420 video needs even sides
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            path = WORK_DIR / f"{self.request.call_id}.png"
            image.save(path, format="PNG")
            return path, np.asarray(image, dtype=np.uint8).copy()

    # -- lifecycle -------------------------------------------------------------------
    def stop(self, reason: str):
        if not self.end_reason:
            self.end_reason = reason
        self.stop_event.set()

    async def run(self):
        from livekit import rtc
        tasks: list[asyncio.Task] = []
        try:
            self.room = room = rtc.Room()
            user_joined = asyncio.Event()

            @room.on("track_subscribed")
            def _track(track, publication, participant):
                if participant.identity == self.request.user_identity and track.kind == rtc.TrackKind.KIND_AUDIO:
                    user_joined.set()
                    tasks.append(asyncio.create_task(self.listen(track)))

            @room.on("participant_disconnected")
            def _left(participant):
                if participant.identity == self.request.user_identity:
                    self.stop("caller_left")

            @room.on("disconnected")
            def _disconnected(*_):
                self.stop("room_closed")

            await room.connect(self.request.livekit_url, self.request.token, rtc.RoomOptions(auto_subscribe=True))
            audio_source = rtc.AudioSource(OUT_RATE, 1, queue_size_ms=200)
            await room.local_participant.publish_track(
                rtc.LocalAudioTrack.create_audio_track("voice", audio_source),
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
            video_source = None
            if self.video:
                h, w = self.idle_rgb.shape[:2]
                video_source = rtc.VideoSource(w, h)
                await room.local_participant.publish_track(
                    rtc.LocalVideoTrack.create_video_track("avatar", video_source),
                    rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA))
                self.av = rtc.AVSynchronizer(audio_source=audio_source, video_source=video_source,
                                             video_fps=self.fps)
            tasks.append(asyncio.create_task(self.media_loop(rtc, audio_source)))
            tasks.append(asyncio.create_task(self.turn_loop()))
            tasks.append(asyncio.create_task(self.watchdog(user_joined)))
            await self.state("listening")
            await self.stop_event.wait()
        except Exception:
            LOG.exception("live_call_failed call_id=%s", self.request.call_id)
            self.end_reason = self.end_reason or "error"
        finally:
            self.finished = True
            self.abort_reply()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await self.state("ended", reason=self.end_reason)
            except Exception:
                pass
            if self.av:
                await self.av.aclose()
            if self.room:
                await self.room.disconnect()
            if self.portrait_path:
                self.portrait_path.unlink(missing_ok=True)
            LOG.info("live_call_ended call_id=%s reason=%s turns=%d", self.request.call_id, self.end_reason, self.seq)

    @property
    def fps(self) -> int:
        import live_ditto
        return live_ditto.OUTPUT_FPS

    async def watchdog(self, user_joined: asyncio.Event):
        try:
            await asyncio.wait_for(user_joined.wait(), JOIN_TIMEOUT_S)
        except asyncio.TimeoutError:
            self.stop("caller_never_joined")
            return
        started = time.monotonic()
        while not self.stop_event.is_set():
            await asyncio.sleep(5)
            now = time.monotonic()
            if now - started > self.request.max_seconds:
                self.stop("max_duration")
            elif not self.speaking and now - self.last_activity > IDLE_TIMEOUT_S:
                self.stop("idle")

    async def state(self, value: str, **extra):
        if self.room is None:
            return
        payload = json.dumps({"type": "state", "state": value, **extra})
        await self.room.local_participant.publish_data(payload, reliable=True, topic="rosely")

    async def event(self, kind: str, **extra):
        if self.room is not None:
            await self.room.local_participant.publish_data(json.dumps({"type": kind, **extra}),
                                                           reliable=True, topic="rosely")

    # -- listening -------------------------------------------------------------------
    async def listen(self, track):
        from livekit import rtc
        import webrtcvad
        vad = webrtcvad.Vad(2)
        detector = TurnDetector(lambda pcm: vad.is_speech(pcm, live_media.VAD_RATE))
        normal_start = detector.start_frames
        stream = rtc.AudioStream(track, sample_rate=live_media.VAD_RATE, num_channels=1,
                                 frame_size_ms=live_media.VAD_FRAME_MS)
        try:
            async for event in stream:
                pcm = bytes(event.frame.data)
                if len(pcm) != live_media.VAD_RATE * live_media.VAD_FRAME_MS // 1000 * 2:
                    continue
                # While the character talks, demand more speech before treating it as the caller.
                detector.start_frames = BARGE_IN_MS // live_media.VAD_FRAME_MS if self.speaking else normal_start
                result = detector.feed(pcm)
                if result == "speech":
                    self.last_activity = time.monotonic()
                    if self.speaking:
                        self.abort_reply()  # barge-in: stop talking, listen
                    await self.state("hearing")
                elif result:
                    self.last_activity = time.monotonic()
                    await self.utterances.put(result[1])
        finally:
            await stream.aclose()

    # -- turns -----------------------------------------------------------------------
    async def turn_loop(self):
        while not self.stop_event.is_set():
            pcm = await self.utterances.get()
            try:
                await self.take_turn(pcm)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("live_turn_failed call_id=%s", self.request.call_id)
                await self.event("error", code="unavailable", message="That reply didn't come through. Please say it again.")
                await self.state("listening")

    async def take_turn(self, pcm: bytes):
        started = time.monotonic()
        samples = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
        wav = live_media.encode_wav(samples, live_media.VAD_RATE)
        await self.state("thinking")
        transcript = await self.transcribe(wav)
        asr_s = time.monotonic() - started
        if len(transcript.strip()) < 2:
            await self.state("listening")
            return
        self.seq += 1
        response = await http.post(self.request.turn_url, headers={"X-Live-Call-Secret": self.request.call_secret},
                                   json={"seq": self.seq, "transcript": transcript,
                                         "audio_base64": base64.b64encode(wav).decode("ascii")})
        if response.status_code >= 400:
            detail = _detail(response)
            await self.event("error", code=detail.get("code", "unavailable"), message=detail.get("message", ""))
            if response.status_code in (402, 403, 404, 410):
                self.stop(detail.get("code", "ended"))  # out of Roses, or the call is over
            else:
                await self.state("listening")
            return
        turn = response.json()
        chat_s = time.monotonic() - started - asr_s
        await self.event("turn", seq=self.seq)
        LOG.info("live_turn call_id=%s seq=%d asr=%.2fs chat=%.2fs", self.request.call_id, self.seq, asr_s, chat_s)
        await self.speak(turn.get("segments") or [turn["reply"]], turn.get("emotion") or "neutral",
                         turn.get("gesture") or "none", started)

    async def transcribe(self, wav: bytes) -> str:
        data = {"language": self.request.language} if self.request.language else {}
        response = await http.post(ASR_URL + "/v1/audio/transcriptions", headers={"X-API-Key": API_KEY},
                                   files={"file": ("speech.wav", wav, "audio/wav")}, data=data)
        response.raise_for_status()
        return response.json().get("text", "")

    async def synthesize(self, text: str) -> np.ndarray:
        response = await http.post(TTS_URL + "/v1/audio/speech", headers={"X-API-Key": API_KEY},
                                   json={"input": text, "voice": self.request.voice_id})
        response.raise_for_status()
        samples, rate = await asyncio.to_thread(live_media.decode_wav, response.content)
        return await asyncio.to_thread(live_media.resample, samples, rate, OUT_RATE)

    async def speak(self, segments: list[str], emotion: str, gesture: str, turn_started: float):
        import live_ditto
        loop = asyncio.get_running_loop()
        reply = ReplyTimeline(self.fps, 0 if not self.video else round(START_BUFFER_S * self.fps))
        reply.turn_started = turn_started
        segments = live_media.quick_start(segments)
        marks: dict[str, float] = {}
        stream = None
        setup = None
        finished = False
        if self.video:
            try:  # an interrupted reply may still be winding down; the renderer serves one reply at a time
                await asyncio.wait_for(self.renderer_idle.wait(), 15)
            except asyncio.TimeoutError:
                LOG.warning("live_render_stuck call_id=%s", self.request.call_id)
            self.renderer_idle.clear()

            def on_done(error):
                if error:
                    LOG.warning("live_render_failed call_id=%s error=%r", self.request.call_id, error)
                reply.render_done = True
                loop.call_soon_threadsafe(self.renderer_idle.set)
            # Ditto setup (~0.4 s) overlaps the first sentence's synthesis.
            setup = asyncio.create_task(asyncio.to_thread(live_ditto.DittoStream, sdk, self.portrait_path, emotion,
                                                          gesture, reply.add_frame, on_done))
        else:
            reply.render_done, reply.last_frame = True, VOICE_ONLY
        self.reply, self.speaking = reply, True
        try:
            for index, text in enumerate(segments):
                if reply.aborted:
                    break
                tts_started = time.monotonic()
                audio = await self.synthesize(text)
                if index == 0:
                    marks["first_tts"] = time.monotonic() - tts_started
                if self.video:
                    if stream is None:
                        try:
                            stream = self.stream = await setup
                        except Exception:
                            self.renderer_idle.set()
                            raise
                        if reply.aborted:
                            break
                    audio = np.pad(audio, (0, live_ditto.pad_to_chunk(len(audio), OUT_RATE)))
                    reply.add_audio(audio)
                    stream.feed(await asyncio.to_thread(
                        live_media.resample, audio, OUT_RATE, live_ditto.DITTO_RATE,
                        len(audio) * live_ditto.DITTO_RATE // OUT_RATE))
                else:
                    reply.add_audio(audio)
                if index == 0:
                    LOG.info("live_first_audio call_id=%s seconds=%.2f marks=%s", self.request.call_id,
                             time.monotonic() - turn_started, {k: round(v, 2) for k, v in marks.items()})
            reply.audio_complete = True
            if stream:
                stream.finish()  # on_done marks the renderer idle once the last frames are out
                finished = True
            while not reply.aborted and reply.played < reply.total_frames():
                await asyncio.sleep(0.05)
        finally:
            reply.audio_complete = True
            if stream is None and setup is not None:
                # Interrupted before the first sentence was fed: release the renderer once it is set up.
                setup.add_done_callback(lambda task: task.cancelled() or task.exception() or task.result().abort())
            elif stream is not None and (reply.aborted or not finished):
                stream.abort()  # returns at once; the renderer frees itself in the background
            self.stream = None
            self.speaking = False
            if self.reply is reply:
                self.reply = None
            if not self.stop_event.is_set():
                await self.state("listening")

    def abort_reply(self):
        if self.reply:
            self.reply.abort()
        if self.stream:
            self.stream.abort()

    # -- media -----------------------------------------------------------------------
    async def media_loop(self, rtc, audio_source):
        """One video frame + its 50 ms of audio per tick; the still portrait and silence when idle."""
        spf = OUT_RATE // self.fps
        silence = rtc.AudioFrame(bytes(spf * 2), OUT_RATE, 1, spf)
        h, w = (self.idle_rgb.shape[:2]) if self.video else (0, 0)

        def video_frame(rgb):
            if rgb.shape[:2] != (h, w):
                import cv2
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            return rtc.VideoFrame(w, h, rtc.VideoBufferType.RGB24, np.ascontiguousarray(rgb, np.uint8).tobytes())

        idle = video_frame(self.idle_rgb) if self.video else None
        fade: list[np.ndarray] = []
        was_playing = False
        while not self.stop_event.is_set():
            reply = self.reply
            item = reply.take() if reply else "end"
            if isinstance(item, tuple):
                rgb, chunk = item
                if not was_playing:
                    was_playing = True
                    LOG.info("live_first_frame call_id=%s seconds=%.2f", self.request.call_id,
                             time.monotonic() - reply.turn_started)
                    await self.state("speaking")
                audio = rtc.AudioFrame(live_media.to_pcm16(chunk).tobytes(), OUT_RATE, 1, spf)
                if self.video:
                    fade = [rgb]
                    await self.av.push(video_frame(rgb))
                    await self.av.push(audio)
                else:
                    await audio_source.capture_frame(audio)
                continue
            if item == "wait" and was_playing:  # next frame still rendering: hold the face, brief pause
                if self.video:
                    await self.av.push(video_frame(fade[0]))
                    await self.av.push(silence)
                else:
                    await audio_source.capture_frame(silence)
                continue
            if item == "end" and was_playing:
                was_playing = False
                if reply and reply.aborted and self.av:
                    await self.av.clear_queue()
                elif fade and self.video:  # ease back to the portrait instead of snapping
                    last = fade[0].astype(np.float32)
                    target = self.idle_rgb.astype(np.float32)
                    for step in range(1, 6):
                        mix = last + (target - last) * (step / 6)
                        await self.av.push(video_frame(mix.astype(np.uint8)))
                        await self.av.push(silence)
                fade = []
            if self.video:
                await self.av.push(idle)
                await self.av.push(silence)
            else:
                await audio_source.capture_frame(silence)


def _detail(response: httpx.Response) -> dict:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return {}
    if isinstance(detail, dict):
        return detail
    return {"message": detail} if isinstance(detail, str) else {}
