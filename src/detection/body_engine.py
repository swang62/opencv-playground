"""MediaPipe Pose + Hand landmarker engine — lazy-loaded BodyEngine."""

from __future__ import annotations

import logging
import ssl
import threading
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src import config

logger = logging.getLogger(__name__)

MODEL_DIR = Path(config.MODELS_DIR)

POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
POSE_FILE = MODEL_DIR / "pose_landmarker_lite.task"

HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_FILE = MODEL_DIR / "hand_landmarker.task"


def _download_model(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    logger.info("Downloading %s ...", path.name)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx) as resp, open(path, "wb") as f:
        f.write(resp.read())
    logger.info("Cached at %s", path)
    return path


class BodyEngine:
    """Lazy-loaded MediaPipe Pose + Hand Landmarker.

    Both use CPU delegate (same GPU crash issue as Face Landmarker on macOS).
    """

    def __init__(self):
        self._pose = None
        self._hand = None
        self._lock = threading.Lock()
        self._frame_count = 0

    def warmup(self):
        self.ensure_pose_loaded()
        self.ensure_hand_loaded()

        dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)

        try:
            self.process_pose(dummy_frame)
        except Exception as exc:
            logger.warning("Pose warmup inference failed: %s", exc)

        try:
            self.process_hands(dummy_frame)
        except Exception as exc:
            logger.warning("Hand warmup inference failed: %s", exc)

    def ensure_pose_loaded(self):
        if self._pose is not None:
            return
        with self._lock:
            if self._pose is not None:
                return
            model_path = _download_model(POSE_URL, POSE_FILE)

            base_opts = python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=python.BaseOptions.Delegate.CPU,
            )
            opts = vision.PoseLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=2,
                output_segmentation_masks=False,
            )
            self._pose = vision.PoseLandmarker.create_from_options(opts)
            logger.info("Pose Landmarker loaded (CPU)")

    def ensure_hand_loaded(self):
        if self._hand is not None:
            return
        with self._lock:
            if self._hand is not None:
                return
            model_path = _download_model(HAND_URL, HAND_FILE)

            base_opts = python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=python.BaseOptions.Delegate.CPU,
            )
            opts = vision.HandLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=4,
            )
            self._hand = vision.HandLandmarker.create_from_options(opts)
            logger.info("Hand Landmarker loaded (CPU)")

    def process_pose(self, frame: np.ndarray) -> list[list[tuple[int, int]]]:
        """Return list of pose landmark lists (one per detected person).

        Each element is a list of 33 (x, y) tuples.
        Returns empty list when no pose detected.
        """
        self.ensure_pose_loaded()
        assert self._pose is not None

        self._frame_count += 1
        timestamp_ms = self._frame_count * 33

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._pose.detect_for_video(mp_img, timestamp_ms)
        if not result.pose_landmarks:
            return []

        h, w = frame.shape[:2]
        poses = []
        for landmarks in result.pose_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            poses.append(pts)
        return poses

    def process_hands(self, frame: np.ndarray) -> list[list[tuple[int, int]]]:
        """Return list of hand landmark lists (one per hand).

        Each element is a list of 21 (x, y) tuples.
        Returns empty list when no hand detected.
        """
        self.ensure_hand_loaded()
        assert self._hand is not None

        self._frame_count += 1
        timestamp_ms = self._frame_count * 33

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._hand.detect_for_video(mp_img, timestamp_ms)
        if not result.hand_landmarks:
            return []

        h, w = frame.shape[:2]
        hands = []
        for landmarks in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            hands.append(pts)
        return hands
