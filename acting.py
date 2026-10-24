"""Emotion + gesture controls for Ditto, chosen by the chat LLM per reply.

emotion: the reply's voice_style (neutral, happy, sad, ...). It sets Ditto's
emotion condition plus a LivePortrait-space expression offset (smile / brows).
gesture: none | nod | shake | tilt | wink, applied as head-pose deltas (degrees)
or an expression offset near the start of the clip.
Unknown values fall back to neutral / none.
"""
import math

import numpy as np

FPS = 25
# Ditto emotion classes: 0 angry, 1 disgust, 2 fear, 3 happy, 4 neutral, 5 sad, 6 surprise, 7 contempt.
EMOTIONS = {  # voice_style: (Ditto emotion labels, smile strength, brow strength)
    "neutral":      ([4],    0.0,   0),
    "calm":         ([4],    0.2,   0),
    "happy":        ([3],    0.9,   4),
    "excited":      ([3, 6], 1.0,  10),
    "playful":      ([3],    0.7,   6),
    "shy":          ([3, 4], 0.5,   3),
    "seductive":    ([3, 7], 0.6,   2),
    "affectionate": ([3],    0.7,   2),
    "sad":          ([5],   -0.2, -12),
    "worried":      ([5, 2], -0.1, -10),
    "angry":        ([0],   -0.3, -18),
    "tired":        ([5, 4], 0.0,  -4),
    "surprised":    ([6],    0.2,  18),
    "confused":     ([4],    0.0,  -8),
    "frustrated":   ([0, 5], -0.2, -14),
}
GESTURES = ("none", "nod", "shake", "tilt", "wink")


# LivePortrait retargeting offsets over 21 keypoints x 3 (flattened like Ditto's exp).
def _add(vec, keypoint, axis, value):
    vec[0, keypoint * 3 + axis] += value


def smile_vec(strength):
    vec = np.zeros((1, 63), np.float32)
    for k, d, c in [(20, 1, -0.01), (14, 1, -0.02), (17, 1, 0.0065), (17, 2, 0.003),
                    (13, 1, -0.00275), (16, 1, -0.00275), (3, 1, -0.0035), (7, 1, -0.0035)]:
        _add(vec, k, d, c * strength)
    return vec


def brow_vec(strength):
    vec = np.zeros((1, 63), np.float32)
    if strength > 0:
        _add(vec, 1, 1, strength * 0.001); _add(vec, 2, 1, strength * -0.001)
    else:
        _add(vec, 1, 0, strength * -0.001); _add(vec, 2, 0, strength * 0.001)
        _add(vec, 1, 1, strength * 0.0003); _add(vec, 2, 1, strength * -0.0003)
    return vec


def wink_vec(strength):
    vec = np.zeros((1, 63), np.float32)
    for k, d, c in [(11, 1, 0.001), (13, 1, -0.0003), (17, 0, 0.0003), (17, 1, 0.0003), (3, 1, -0.0003)]:
        _add(vec, k, d, c * strength)
    return vec


def envelope(n, ramp):
    """0 -> 1 -> 0 so controls ease in and out instead of popping."""
    env = np.ones(n, np.float32)
    r = min(ramp, n // 2)
    if r:
        env[:r] = np.linspace(0, 1, r)
        env[-r:] = np.linspace(1, 0, r)
    return env


def controls(emotion, gesture, num_frames):
    """Return (Ditto emo labels, per-frame ctrl_info) for one spoken clip."""
    labels, smile, brow = EMOTIONS.get(emotion, EMOTIONS["neutral"])
    gesture = gesture if gesture in GESTURES else "none"
    n = max(int(num_frames), 1)
    exp = envelope(n, 8)[:, None, None] * (smile_vec(smile) + brow_vec(brow))
    pose = {key: np.zeros(n, np.float32) for key in ("delta_pitch", "delta_yaw", "delta_roll")}
    t = np.arange(n) / FPS
    if gesture == "nod":        # two gentle nods over ~1.2 s
        m = min(30, n)
        pose["delta_pitch"][:m] += 6 * np.sin(2 * math.pi * t[:m] / 0.6) * envelope(m, 4)
    elif gesture == "shake":    # side-to-side "no"
        m = min(36, n)
        pose["delta_yaw"][:m] += 9 * np.sin(2 * math.pi * t[:m] / 0.6) * envelope(m, 4)
    elif gesture == "tilt":     # held tilt for the whole reply
        env = envelope(n, 10)
        pose["delta_roll"] += 7 * env
        pose["delta_pitch"] += -3 * env
    elif gesture == "wink":     # quick wink shortly after the clip starts
        start = min(8, n - 1)
        m = min(10, n - start)
        exp[start:start + m] += envelope(m, 3)[:, None, None] * wink_vec(30)
    ctrl = {}
    for i in range(n):
        frame = {key: float(values[i]) for key, values in pose.items() if abs(values[i]) > 1e-4}
        if np.abs(exp[i]).sum() > 0:
            frame["delta_exp"] = exp[i]
        if frame:
            ctrl[i] = frame
    return labels, ctrl
