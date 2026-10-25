"""Ditto online (streaming) rendering for live calls.

One DittoStream per spoken reply: sentence audio is fed as it is synthesized and
frames come back while the rest is still rendering. On the 3090 the face decoder
caps Ditto at ~24 fps, just under real time, so every fifth frame is dropped
before the GPU-heavy warp/decode stages: 20 fps output with headroom.

Timeline: frame k (at OUTPUT_FPS) shows the face for reply audio
[k / OUTPUT_FPS, (k + 1) / OUTPUT_FPS). Each fed piece is padded with silence to a
whole online chunk (CHUNK_SAMPLES), and the caller pads its playback audio the same
way (pad_to_chunk), so lips and sound never drift apart across sentences.
"""
from __future__ import annotations

import queue
import tempfile
import threading
from pathlib import Path
from typing import Callable

import numpy as np

import acting

DITTO_RATE = 16000
DITTO_FPS = 25
OUTPUT_FPS = 20
KEEP = DITTO_FPS // (DITTO_FPS - OUTPUT_FPS)  # keep 4 of every 5 frames
CHUNKSIZE = (3, 5, 2)  # online HuBERT window in frames: past, current, future
FRAME_SAMPLES = DITTO_RATE // DITTO_FPS  # 640
CHUNK_SAMPLES = CHUNKSIZE[1] * FRAME_SAMPLES  # 3200 = 0.2 s of new audio per chunk
CONTEXT_SAMPLES = CHUNKSIZE[0] * FRAME_SAMPLES  # 1920 of leading context
WINDOW_SAMPLES = sum(CHUNKSIZE) * FRAME_SAMPLES + 80  # 6480, as upstream inference.py
MAX_REPLY_FRAMES = DITTO_FPS * 120  # emotion/gesture controls cover replies up to 2 minutes


def pad_to_chunk(samples: int, rate: int) -> int:
    """Silence (in samples at `rate`) that rounds a piece up to a whole online chunk."""
    chunk = rate * CHUNK_SAMPLES // DITTO_RATE
    return (-samples) % chunk


def kept(index: int) -> bool:
    """Whether 25 fps frame `index` survives the 25 -> 20 fps drop."""
    return index % KEEP != KEEP - 1 if KEEP > 1 else True


class _Sink:
    """Stands in for Ditto's MP4 writer; hands each RGB frame to the caller."""

    def __init__(self, on_frame: Callable[[np.ndarray], None]):
        self.on_frame = on_frame

    def __call__(self, frame_rgb, fmt="rgb"):
        self.on_frame(frame_rgb)

    def close(self):
        pass


class DittoStream:
    """Feed reply audio piece by piece; frames arrive through on_frame (worker thread).

    on_done(error) runs once rendering ends (after finish(), abort() or a failure).
    """

    def __init__(self, sdk, source_path: Path, emotion: str, gesture: str,
                 on_frame: Callable[[np.ndarray], None], on_done: Callable[[Exception | None], None]):
        self.sdk, self.on_done = sdk, on_done
        labels, ctrl = acting.controls(emotion, gesture, MAX_REPLY_FRAMES)
        # The output path is never written (frames go to on_frame); keep it out of the shared clip store.
        sdk.setup(str(source_path), str(Path(tempfile.gettempdir()) / "ditto-live.mp4"), online_mode=True, emo=labels)
        sdk.setup_Nd(N_d=MAX_REPLY_FRAMES, ctrl_info=ctrl)
        sdk.writer = _Sink(on_frame)
        put, count = sdk.warp_f3d_queue.put, [0]

        def dropping_put(item, *args, **kwargs):
            if item is not None:
                count[0] += 1
                if not kept(count[0] - 1):
                    return None  # skipped before warp/decode, the stages that set the frame rate
            return put(item, *args, **kwargs)

        sdk.warp_f3d_queue.put = dropping_put
        self.audio = np.zeros(CONTEXT_SAMPLES, np.float32)  # leading context, as upstream
        self.pos = 0
        self.pending: queue.Queue = queue.Queue()
        self.aborted = False
        self.thread = threading.Thread(target=self._run, name="ditto-live", daemon=True)
        self.thread.start()

    def feed(self, audio16k: np.ndarray) -> None:
        """Queue a piece of 16 kHz mono float audio; its length must be a whole chunk."""
        if len(audio16k) % CHUNK_SAMPLES:
            raise ValueError("pad live audio to a whole chunk first (pad_to_chunk)")
        self.pending.put(np.asarray(audio16k, np.float32))

    def finish(self) -> None:
        self.pending.put(None)

    def abort(self) -> None:
        """Stop now (barge-in); frames still in flight are discarded by the caller."""
        self.aborted = True
        self.sdk.stop_event.set()
        self.pending.put(None)

    def _run(self) -> None:
        error = None
        try:
            while not self.aborted:
                piece = self.pending.get()
                if piece is None:
                    break
                self.audio = np.concatenate([self.audio, piece])
                # Chunk at pos renders audio [pos, pos + CHUNK) of the reply (after the context pad).
                while not self.aborted and self.pos + CONTEXT_SAMPLES + CHUNK_SAMPLES <= len(self.audio):
                    window = self.audio[self.pos:self.pos + WINDOW_SAMPLES]
                    if len(window) < WINDOW_SAMPLES:  # future context not synthesized yet
                        window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))
                    self.sdk.run_chunk(window, CHUNKSIZE)
                    self.pos += CHUNK_SAMPLES
                # Older audio is no longer needed for future windows.
                if self.pos > CHUNK_SAMPLES * 4:
                    drop = self.pos - CHUNK_SAMPLES
                    self.audio, self.pos = self.audio[drop:], self.pos - drop
            self.sdk.close()  # re-raises a pipeline worker's exception
        except Exception as exc:  # surfaced to the caller; the call itself keeps going
            error = exc
        finally:
            self.on_done(error)  # reported even after abort: a stall is often a worker failure
