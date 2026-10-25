"""Pure media helpers for live calls (no GPU, no LiveKit): turn detection, audio
conversion, and the reply timeline that pairs each video frame with its audio.

Kept separate from live_agent.py so it is unit-tested without models.
"""
from __future__ import annotations

import io
import threading
from collections import deque

import numpy as np
import soundfile as sf

OUT_RATE = 24000  # character audio published to the call (Chatterbox's native rate)
VAD_RATE = 16000
VAD_FRAME_MS = 30


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio.mean(axis=1), rate


def encode_wav(samples: np.ndarray, rate: int) -> bytes:
    out = io.BytesIO()
    sf.write(out, samples, rate, format="WAV", subtype="PCM_16")
    return out.getvalue()


def resample(samples: np.ndarray, src: int, dst: int, length: int | None = None) -> np.ndarray:
    """Resample mono float audio; `length` forces an exact output length (pad/trim)."""
    if src != dst and len(samples):
        import soxr
        samples = soxr.resample(samples, src, dst)
    samples = np.asarray(samples, np.float32)
    if length is not None:
        samples = samples[:length] if len(samples) >= length else np.pad(samples, (0, length - len(samples)))
    return samples


def to_pcm16(samples: np.ndarray) -> np.ndarray:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)


def quick_start(segments: list[str], max_words: int = 10, min_head: int = 4) -> list[str]:
    """Split a long opening sentence at an early comma so the first words synthesize sooner.

    The head keeps at least min_head words: it must last long enough to cover the
    next piece's synthesis (~0.65 s per spoken second), or the pause would show.
    """
    if not segments:
        return segments
    words = segments[0].split()
    if len(words) <= max_words:
        return segments
    for i, word in enumerate(words[:max_words], 1):
        if i >= min_head and word.endswith((",", ";", ":", "\u2014", "-")):
            return [" ".join(words[:i]), " ".join(words[i:]), *segments[1:]]
    return segments


class TurnDetector:
    """Voice activity -> turns, from 30 ms 16 kHz int16 microphone frames.

    feed() returns "speech" once the speaker has talked for start_ms (used for
    barge-in and the "hearing you" state), then ("utterance", pcm) after end_ms of
    silence. The utterance includes preroll_ms before speech began and is capped
    at max_ms (the backend accepts at most 15 s per turn).
    """

    def __init__(self, is_speech, *, start_ms=240, end_ms=700, preroll_ms=300, max_ms=14500):
        self.is_speech = is_speech  # (pcm_bytes) -> bool, e.g. webrtcvad.Vad(2).is_speech at 16 kHz
        self.start_frames = max(start_ms // VAD_FRAME_MS, 1)
        self.end_frames = max(end_ms // VAD_FRAME_MS, 1)
        self.max_frames = max_ms // VAD_FRAME_MS
        self.preroll: deque[bytes] = deque(maxlen=max(preroll_ms // VAD_FRAME_MS, 1))
        self.reset()

    def reset(self):
        self.frames: list[bytes] = []
        self.voiced_run = 0
        self.silent_run = 0
        self.speaking = False
        self.preroll.clear()

    def feed(self, pcm: bytes):
        voiced = self.is_speech(pcm)
        if not self.speaking:
            self.preroll.append(pcm)
            self.voiced_run = self.voiced_run + 1 if voiced else 0
            if self.voiced_run >= self.start_frames:
                self.speaking, self.silent_run = True, 0
                self.frames = list(self.preroll)
                return "speech"
            return None
        self.frames.append(pcm)
        self.silent_run = 0 if voiced else self.silent_run + 1
        if self.silent_run >= self.end_frames or len(self.frames) >= self.max_frames:
            pcm_all = b"".join(self.frames[:len(self.frames) - min(self.silent_run, len(self.frames)) + 3])
            self.reset()
            return ("utterance", pcm_all)
        return None


class ReplyTimeline:
    """One spoken reply: frames (from Ditto's thread) and audio, paired by index.

    Frame k shows the face for audio samples [k * spf, (k + 1) * spf) at OUT_RATE.
    take() returns the next (frame, audio) pair, "wait" while the next frame is
    still rendering, or "end" once all audio has been played.
    """

    def __init__(self, fps: int, start_buffer_frames: int):
        self.spf = OUT_RATE // fps
        self.start_buffer = start_buffer_frames
        self.frames: deque = deque()
        self.audio = np.zeros(0, np.float32)
        self.played = 0  # frames handed out
        self.lock = threading.Lock()
        self.audio_complete = False  # every sentence's audio has been added
        self.render_done = False  # Ditto finished (or failed)
        self.aborted = False
        self.started = False
        self.last_frame = None

    def add_frame(self, frame) -> None:  # called from Ditto's writer thread
        with self.lock:
            if not self.aborted:
                self.frames.append(frame)

    def add_audio(self, samples: np.ndarray) -> None:
        with self.lock:
            self.audio = np.concatenate([self.audio, samples.astype(np.float32)])

    def total_frames(self) -> int:
        return -(-len(self.audio) // self.spf)

    def take(self):
        with self.lock:
            if self.aborted:
                return "end"
            if self.audio_complete and self.played >= self.total_frames():
                return "end"
            if not self.started:
                # Start once a small buffer is ready, so Ditto's slight jitter never stalls playback.
                ready = len(self.frames) >= self.start_buffer or (self.render_done and self.frames)
                if not ready:
                    return "end" if self.render_done and self.audio_complete else "wait"
                self.started = True
            start = self.played * self.spf
            chunk = self.audio[start:start + self.spf]
            if len(chunk) < self.spf:
                if not self.audio_complete:
                    return "wait"  # this frame's audio is still being synthesized
                chunk = np.pad(chunk, (0, self.spf - len(chunk)))
            if self.frames:
                self.last_frame = self.frames.popleft()
            elif not self.render_done or self.last_frame is None:
                return "wait"
            # else: the renderer stopped early; finish the audio on the last face.
            self.played += 1
            return self.last_frame, chunk

    def abort(self) -> None:
        with self.lock:
            self.aborted = True
            self.frames.clear()
