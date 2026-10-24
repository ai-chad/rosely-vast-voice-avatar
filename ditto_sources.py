"""Motion-clip sources for Ditto: a shared clip store plus cached avatar registration.

A motion clip is a short pre-generated video of the character (body, hands, head);
Ditto keeps that motion and drives only the mouth and expression from the reply
audio. Registering a 10 s clip takes ~6 s and ~2.3 GB (per-frame feature volumes),
so registrations are cached in RAM (few clips) and on disk (features as float16).

Clips live in DITTO_CLIP_DIR as <sha256>.mp4; the gateway writes them, Ditto reads them.
"""
from __future__ import annotations

import copy
import hashlib
import logging
import os
import pickle
import re
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

LOG = logging.getLogger("uvicorn.error")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ALL_FRAMES = 100_000  # register the whole clip so one entry serves replies of any length
# Per-frame values that Ditto's setup may replace or adjust; big arrays are shared read-only.
SMALL_KEYS = ("x_s_info_lst", "M_c2o_lst", "eye_open_lst", "eye_ball_lst", "sc")


def _media_root() -> Path:
    return Path(os.getenv("AVATAR_MEDIA_ROOT", "/workspace/rosely-avatar-data"))


def clip_dir() -> Path:
    return Path(os.getenv("DITTO_CLIP_DIR", str(_media_root() / "ditto-clips")))


def clip_path(sha256: str) -> Path:
    if not SHA_RE.fullmatch(sha256 or ""):
        raise ValueError("motion clip id must be a lowercase sha256")
    return clip_dir() / f"{sha256}.mp4"


def store_clip(data: bytes, sha256: str) -> Path:
    """Write a clip under its digest after verifying it (atomic; safe to repeat)."""
    path = clip_path(sha256)
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("motion clip sha256 mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
    partial.write_bytes(data)
    os.replace(partial, path)
    return path


def _fresh(info: dict) -> dict:
    """Copy the parts Ditto may modify; share the large feature and frame arrays."""
    out = dict(info)
    for key in SMALL_KEYS:
        if key in out:
            out[key] = copy.deepcopy(out[key])
    return out


class RegistrationCache:
    """Drop-in wrapper for StreamSDK.avatar_registrar that caches motion-clip registrations."""

    def __init__(self, registrar, *, ram_items: int = 3, disk_dir: Path | None = None,
                 disk_bytes: float = 20e9):
        self.registrar = registrar
        self.ram_items = max(int(ram_items), 0)
        self.disk_dir = Path(disk_dir) if disk_dir else None
        self.disk_bytes = disk_bytes
        self.ram: OrderedDict[str, dict] = OrderedDict()
        self.lock = threading.Lock()
        self.last = None  # "ram" | "disk" | "registered" | "uncached" for the latest call

    def __call__(self, source_path, max_dim=1920, n_frames=-1, **crop):
        path = Path(source_path)
        sha = path.stem
        if path.parent != clip_dir() or path.suffix != ".mp4" or not SHA_RE.fullmatch(sha):
            self.last = "uncached"
            return self.registrar(source_path, max_dim=max_dim, n_frames=n_frames, **crop)
        key = f"{sha}-{max_dim}-" + "-".join(f"{k}={crop[k]}" for k in sorted(crop))
        with self.lock:
            if key in self.ram:
                self.ram.move_to_end(key)
                self.last = "ram"
                return _fresh(self.ram[key])
        info = self._load_disk(key)
        if info is not None:
            self.last = "disk"
        else:
            info = self.registrar(source_path, max_dim=max_dim, n_frames=ALL_FRAMES, **crop)
            self.last = "registered"
            if self.disk_dir is not None:
                threading.Thread(target=self._save_disk, args=(key, info), daemon=True).start()
        if self.ram_items:
            with self.lock:
                self.ram[key] = info
                self.ram.move_to_end(key)
                while len(self.ram) > self.ram_items:
                    self.ram.popitem(last=False)
        return _fresh(info)

    def stats(self) -> dict:
        disk = sorted(self.disk_dir.glob("*.pkl")) if self.disk_dir and self.disk_dir.exists() else []
        return {"ram_items": len(self.ram), "disk_items": len(disk),
                "disk_gb": round(sum(p.stat().st_size for p in disk) / 1e9, 2)}

    def _disk_file(self, key: str) -> Path:
        return self.disk_dir / f"{key}.pkl"

    def _load_disk(self, key: str) -> dict | None:
        if self.disk_dir is None or not self._disk_file(key).exists():
            return None
        try:
            with open(self._disk_file(key), "rb") as handle:
                info = pickle.load(handle)
            os.utime(self._disk_file(key))  # mark as recently used for eviction
        except Exception:
            LOG.warning("Discarding unreadable Ditto registration cache entry %s", key[:16])
            self._disk_file(key).unlink(missing_ok=True)
            return None
        info["f_s_lst"] = [np.asarray(f, dtype=np.float32) for f in info["f_s_lst"]]
        return info

    def _save_disk(self, key: str, info: dict) -> None:
        try:
            self.disk_dir.mkdir(parents=True, exist_ok=True)
            stored = dict(info)
            stored["f_s_lst"] = [f.astype(np.float16) for f in info["f_s_lst"]]  # halves the size
            partial = self.disk_dir / f"{key}.{os.getpid()}.part"
            with open(partial, "wb") as handle:
                pickle.dump(stored, handle, protocol=5)
            os.replace(partial, self._disk_file(key))
            self._evict_disk()
        except Exception:
            LOG.exception("Could not persist Ditto registration %s", key[:16])

    def _evict_disk(self) -> None:
        entries = sorted(self.disk_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in entries)
        while entries and total > self.disk_bytes:
            oldest = entries.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)
