"""Stream a YouTube video via yt-dlp with a frame buffer for smooth playback.

Resolves a YouTube URL to a direct stream URL using the system ``yt-dlp``
binary (which picks up your ``~/.yt-dlp.conf`` cookies and VPN config),
then wraps it in ``cv2.VideoCapture`` with a background reader thread
and a frame deque to absorb network jitter.  No seek, no progress —
behaves like a live webcam for the pipeline.
"""

from __future__ import annotations

import collections
import logging
import subprocess
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BUFFER_SIZE = 300
STREAM_RECONNECT_DELAY = 1.0


class YouTubeSource:
    """Streams a YouTube video as a drop-in camera replacement.

    Implements ``.read() -> (bool, frame)``, ``.release()``, ``.isOpened()``.

    A daemon thread fills a frame deque at the stream's native frame rate,
    smoothing out network jitter from the underlying stream.

    No ``.seek()`` — the stream is live-like; the seek bar stays hidden.
    """

    def __init__(self, url: str, target_size: tuple[int, int] | None = None):
        self._url = url
        self._target_size = target_size
        self._buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=BUFFER_SIZE
        )
        self._buffer_lock = threading.Lock()
        self._stopped = False

        stream_url = self._resolve(url)
        self._cap = cv2.VideoCapture(stream_url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open YouTube stream: {url}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_interval = 1.0 / fps if fps > 0 else 1.0 / 30.0
        logger.info(
            "YouTube stream FPS: %s (interval=%.3fs)",
            fps or "auto",
            self._frame_interval,
        )

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    @property
    def url(self) -> str:
        return self._url

    @staticmethod
    def _resolve(url: str) -> str:
        """Resolve a YouTube URL to a direct stream URL via the system yt-dlp."""
        try:
            result = subprocess.run(
                ["yt-dlp", "-g", "--no-warnings", url],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "yt-dlp not found. Install it with `brew install yt-dlp`"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("yt-dlp timed out resolving the URL")

        if result.returncode != 0:
            msg = result.stderr.strip() or "unknown error"
            raise RuntimeError(f"yt-dlp failed: {msg}")

        lines = result.stdout.strip().splitlines()
        if not lines:
            raise RuntimeError("No stream URL returned by yt-dlp")

        stream_url = lines[0]
        logger.info("YouTube stream resolved: %s …", stream_url[:80])
        return stream_url

    def _reader_loop(self):
        """Background thread: fill the frame deque at the stream's native FPS."""
        last_read = 0.0
        while not self._stopped:
            now = time.time()
            if last_read > 0:
                elapsed = now - last_read
                if elapsed < self._frame_interval:
                    time.sleep(self._frame_interval - elapsed)

            ret, frame = self._cap.read()
            if not ret:
                logger.debug("YouTube stream read failed, reconnecting…")
                if not self._stopped:
                    time.sleep(STREAM_RECONNECT_DELAY)
                continue

            last_read = time.time()

            if self._target_size is not None:
                frame = self._letterbox(frame, self._target_size)
            with self._buffer_lock:
                self._buffer.append(frame)

    def read(self):
        with self._buffer_lock:
            if self._buffer:
                return True, self._buffer.popleft()
            return False, None

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
        self._stopped = True
        self._reader_thread.join(timeout=2)
        self._cap.release()

    def isOpened(self):
        return self._cap.isOpened()

    def get(self, prop):
        return self._cap.get(prop)
