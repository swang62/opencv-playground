"""Video file player that acts as a drop-in camera replacement with infinite looping."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class VideoFilePlayer:
    """Wraps cv2.VideoCapture for a video file, cycling infinitely on end-of-file.

    Implements the same interface the pipeline expects:
    ``.read() -> (bool, frame | None)``, ``.release()``, ``.isOpened()``.

    FPS is throttled to the video's original frame rate so the capture
    loop doesn't burn through all frames instantly.

    When *target_size* ``(width, height)`` is given, frames are
    letterboxed (resized + padded with black bars) to that exact
    resolution while preserving the original aspect ratio.
    """

    def __init__(self, path: str, target_size: tuple[int, int] | None = None):
        self._path = path
        self._target_size = target_size
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_interval = 1.0 / self._fps
        self._last_read_time = 0.0
        self._cap_lock = threading.Lock()

    @property
    def path(self) -> str:
        return self._path

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration(self) -> float:
        return self._frame_count / self._fps if self._fps > 0 else 0.0

    @property
    def progress(self) -> float:
        if self._frame_count == 0:
            return 0.0
        cur = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        return cur / self._frame_count

    @property
    def current_time(self) -> float:
        cur = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        return cur / self._fps if self._fps > 0 else 0.0

    def seek(self, position: float):
        """Seek to *position* as a fraction (0.0 = start, 1.0 = end)."""
        with self._cap_lock:
            frame_idx = int(position * self._frame_count)
            self._cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                max(0, min(frame_idx, self._frame_count - 1)),
            )
            self._last_read_time = 0.0

    def read(self):
        with self._cap_lock:
            now = time.time()
            elapsed = now - self._last_read_time
            if elapsed < self._frame_interval and self._last_read_time > 0:
                time.sleep(self._frame_interval - elapsed)

            ret, frame = self._cap.read()
            if not ret:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._last_read_time = 0.0
                ret, frame = self._cap.read()
                if not ret:
                    return False, None

            self._last_read_time = time.time()

        if self._target_size is not None:
            frame = self._letterbox(frame, self._target_size)

        return True, frame

    def _letterbox(self, frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        h, w = frame.shape[:2]
        tw, th = target_size

        scale = min(tw / w, th / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        x_off = (tw - new_w) // 2
        y_off = (th - new_h) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        return canvas

    def release(self):
        self._cap.release()

    def isOpened(self):
        return self._cap.isOpened()

    def get(self, prop):
        return self._cap.get(prop)
