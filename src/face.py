"""MediaPipe Face Landmarker — 478-point 3D face mesh for Apple Silicon."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "face_landmarker_v2.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# ---------------------------------------------------------------------------
# Face mesh connections — these define the 478-point wireframe topology.
# Indices are from the MediaPipe Face Mesh canonical model.
# Grouped by facial feature for a clean wireframe overlay.
# ---------------------------------------------------------------------------

_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109, 10,
]

_LEFT_EYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160,
    161, 246, 33,
]

_RIGHT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,
    384, 398, 362,
]

_LEFT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
_RIGHT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]

_NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 45]
_NOSE_BOTTOM = [2, 97, 98, 327, 326, 45]

_LIPS_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
    0, 37, 39, 40, 185, 61,
]

_LIPS_INNER = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 78,
]

_ALL_CONTOURS = [
    _FACE_OVAL, _LEFT_EYE, _RIGHT_EYE, _LEFT_BROW, _RIGHT_BROW,
    _NOSE_BRIDGE, _NOSE_BOTTOM, _LIPS_OUTER, _LIPS_INNER,
]

# Build flattened list of (i, j) index pairs for OpenCV polylines.
_FACE_CONNECTIONS: list[np.ndarray] = []
for contour in _ALL_CONTOURS:
    pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 1)
    _FACE_CONNECTIONS.append(pts)


def _download_model() -> Path:
    """Download Face Landmarker model if not cached, return path."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_FILE.exists():
        return MODEL_FILE

    import ssl
    import urllib.request

    logger.info("Downloading Face Landmarker model (~12 MB) ...")
    # macOS Python framework build often lacks root CA bundle.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(MODEL_URL, context=ctx) as resp:
        with open(MODEL_FILE, "wb") as f:
            f.write(resp.read())
    logger.info("Face Landmarker model cached at %s", MODEL_FILE)
    return MODEL_FILE


# ---------------------------------------------------------------------------
# Face drawing helpers
# ---------------------------------------------------------------------------

_FPS_COLOR = (0, 255, 0)  # green

# Per-contour colors in BGR. Order matches _ALL_CONTOURS.
_CONTOUR_COLORS = [
    (0, 255, 0),          # _FACE_OVAL - pure green outline
    (255, 255, 0),        # _LEFT_EYE - cyan
    (255, 255, 0),        # _RIGHT_EYE - cyan
    (0, 255, 100),        # _LEFT_BROW - warm green
    (0, 255, 100),        # _RIGHT_BROW - warm green
    (100, 255, 0),        # _NOSE_BRIDGE - yellow-green
    (100, 255, 0),        # _NOSE_BOTTOM - yellow-green
    (0, 150, 255),        # _LIPS_OUTER - orange
    (0, 150, 255),        # _LIPS_INNER - orange
]

# Precompute a color per landmark index (0-477) based on contour membership.
_LANDMARK_COLORS: list[tuple[int, int, int]] = []
for i in range(478):
    _c = (0, 200, 0)  # default medium green
    for ci, contour in enumerate(_ALL_CONTOURS):
        if i in contour:
            _c = _CONTOUR_COLORS[ci]
            break
    _LANDMARK_COLORS.append(_c)


# Only draw every Nth point to keep the overlay clean (and fast).
_POINT_STRIDE = 4


def draw_face_mesh(frame: np.ndarray, faces, fps: float) -> np.ndarray:
    """Draw the 478-point mesh with per-feature colors for each detected face.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame (modified in-place for speed).
    faces : list[dict]
        Each dict must have ``"landmarks"`` (list of (x, y) tuples, length 478).
    fps : float
        Current frames-per-second to overlay.

    Returns
    -------
    np.ndarray
        The annotated frame.
    """
    for face in faces:
        pts = face["landmarks"]

        # -- wireframe connections (per-contour colors) --
        for ci, conn in enumerate(_FACE_CONNECTIONS):
            color = _CONTOUR_COLORS[ci]
            pixel_pts = np.array(
                [(pts[i][0], pts[i][1]) for i in conn.flatten() if i < len(pts)],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            cv2.polylines(frame, [pixel_pts], isClosed=False, color=color, thickness=2)

        # -- landmark dots (colored by feature) --
        for i in range(0, len(pts), _POINT_STRIDE):
            cv2.circle(
                frame, (pts[i][0], pts[i][1]), 2,
                _LANDMARK_COLORS[i], -1, lineType=cv2.LINE_AA,
            )

    # -- FPS overlay --
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, _FPS_COLOR, 2,
    )

    return frame


# ---------------------------------------------------------------------------
# FaceEngine — lazy-loaded MediaPipe Face Landmarker
# ---------------------------------------------------------------------------


class FaceEngine:
    """Lazy-loaded MediaPipe Face Landmarker for 478-point 3D face mesh.

    Downloads the model on first use. Uses the GPU (Metal) delegate on Apple
    Silicon for the best performance (30-50 FPS on M1).
    """

    def __init__(self):
        self._landmarker = None
        self._lock = threading.Lock()
        self._frame_count = 0

    def _ensure_loaded(self):
        if self._landmarker is not None:
            return
        with self._lock:
            if self._landmarker is not None:
                return

            model_path = _download_model()

            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import mediapipe as mp  # noqa: F401 — needed for mp.Image below

            # Try GPU (Metal) first; fall back to CPU if not available.
            try:
                base_opts = python.BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=python.BaseOptions.Delegate.GPU,
                )
                opts = vision.FaceLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=vision.RunningMode.VIDEO,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=5,
                )
                self._landmarker = vision.FaceLandmarker.create_from_options(opts)
                logger.info("Face Landmarker loaded (GPU delegate)")
            except RuntimeError:
                logger.info("GPU delegate unavailable, falling back to CPU")
                base_opts = python.BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=python.BaseOptions.Delegate.CPU,
                )
                opts = vision.FaceLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=vision.RunningMode.VIDEO,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=5,
                )
                self._landmarker = vision.FaceLandmarker.create_from_options(opts)
                logger.info("Face Landmarker loaded (CPU)")

    def process(self, frame: np.ndarray):
        """Run face landmark detection on *frame*.

        Returns a list of dicts, one per detected face::

            {
                "label": "Face 1",         # placeholder label
                "confidence": 0.95,        # fixed, MediaPipe confidence
                "bbox": (x1, y1, x2, y2),  # pixel coords
                "landmarks": [(x, y), ...], # 478 (x, y) tuples
            }

        Returns an empty list when no face is detected.
        """
        self._ensure_loaded()
        assert self._landmarker is not None  # _ensure_loaded guarantees this

        self._frame_count += 1
        timestamp_ms = self._frame_count * 33  # ~30 FPS pacing

        import mediapipe as mp

        # GPU (Metal) delegate requires 4-channel SRGBA; CPU accepts both.
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)

        result = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        faces = []
        if not result.face_landmarks:
            return faces

        h, w = frame.shape[:2]
        for landmarks in result.face_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            # bounding box from landmark extremes
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1 = max(0, min(xs)), max(0, min(ys))
            x2, y2 = min(w - 1, max(xs)), min(h - 1, max(ys))

            faces.append({
                "label": "Face",
                "confidence": 0.95,
                "bbox": (x1, y1, x2, y2),
                "landmarks": pts,
            })

        return faces
