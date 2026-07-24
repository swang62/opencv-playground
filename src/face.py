"""MediaPipe Face Landmarker — 478-point 3D face mesh for Apple Silicon."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from src import config

logger = logging.getLogger(__name__)

MODEL_DIR = Path(config.MODELS_DIR)
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

FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
    10,
]

LEFT_EYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
    33,
]

RIGHT_EYE = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
    362,
]

LEFT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
RIGHT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]

NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 45]
NOSE_BOTTOM = [2, 97, 98, 327, 326, 45]

LIPS_OUTER = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
    61,
]

LIPS_INNER = [
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
    324,
    318,
    402,
    317,
    14,
    87,
    178,
    88,
    95,
    78,
]

ALL_CONTOURS = [
    FACE_OVAL,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_BROW,
    RIGHT_BROW,
    NOSE_BRIDGE,
    NOSE_BOTTOM,
    LIPS_OUTER,
    LIPS_INNER,
]

# Build flattened list of (i, j) index pairs for OpenCV polylines.
FACE_CONNECTIONS = []
for contour in ALL_CONTOURS:
    pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 1)
    FACE_CONNECTIONS.append(pts)


def download_model() -> Path:
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

# Per-contour colors in BGR. Order matches ALL_CONTOURS.
CONTOUR_COLORS = [
    (0, 255, 0),  # FACE_OVAL - pure green outline
    (255, 255, 0),  # LEFT_EYE - cyan
    (255, 255, 0),  # RIGHT_EYE - cyan
    (0, 255, 100),  # LEFT_BROW - warm green
    (0, 255, 100),  # RIGHT_BROW - warm green
    (100, 255, 0),  # NOSE_BRIDGE - yellow-green
    (100, 255, 0),  # NOSE_BOTTOM - yellow-green
    (0, 150, 255),  # LIPS_OUTER - orange
    (0, 150, 255),  # LIPS_INNER - orange
]

# Precompute a color per landmark index (0-477) — all the same green.
LANDMARK_COLORS = [config.OVERLAY_COLOR] * 478


def draw_face_mesh(
    frame: np.ndarray,
    faces,
    show_wireframe: bool = True,
    show_headpose: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw face mesh, head pose arrow, and attribute labels on *frame*.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame (modified in-place for speed).
    faces : list[dict]
        Each dict has ``"landmarks"`` (478 (x, y) tuples), ``"bbox"``, and
        optional ``"headpose"`` (pitch, yaw, roll), ``"age"``, ``"gender"``,
        ``"emotion"`` keys.
    fps : float
        Current frames-per-second to overlay.
    show_wireframe : bool
        Whether to draw the 478-point mesh wireframe.
    show_headpose : bool
        Whether to draw the head pose direction arrow.
    show_labels : bool
        Whether to draw age/gender/emotion text.
    """
    for face in faces:
        pts = face["landmarks"]
        bbox = face.get("bbox", (0, 0, 0, 0))
        x1, y1, x2, y2 = map(int, bbox)

        if show_wireframe:
            for ci, conn in enumerate(FACE_CONNECTIONS):
                pixel_pts = np.array(
                    [(pts[i][0], pts[i][1]) for i in conn.flatten() if i < len(pts)],
                    dtype=np.int32,
                ).reshape(-1, 1, 2)
                cv2.polylines(
                    frame,
                    [pixel_pts],
                    isClosed=False,
                    color=config.OVERLAY_COLOR,
                    thickness=config.OVERLAY_THICKNESS,
                )

            for i in range(0, len(pts), config.FACE_POINT_STRIDE):
                cv2.circle(
                    frame,
                    (pts[i][0], pts[i][1]),
                    2,
                    LANDMARK_COLORS[i],
                    -1,
                    lineType=cv2.LINE_AA,
                )

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        face_w = max(x2 - x1, 1)

        if show_headpose and "headpose" in face:
            hp = face["headpose"]

            # Direction arrow
            arrow_len = min(face_w * 0.8, 120)
            dy = -arrow_len * np.sin(np.radians(hp["pitch"]))
            dx = -arrow_len * np.sin(np.radians(hp["yaw"]))
            ex = int(cx + dx)
            ey = int(cy + dy)
            cv2.arrowedLine(
                frame,
                (cx, cy),
                (ex, ey),
                (0, 200, 255),
                5,
                cv2.LINE_AA,
                tipLength=0.25,
            )

        if show_labels:
            parts = []
            if "age" in face:
                parts.append(f"Age: {face['age']}")
            if "gender" in face:
                parts.append(f"Gender: {face['gender']}")
            if "emotion" in face:
                em = face["emotion"]
                if em in ("Neutral", "neutral"):
                    em = "-"
                parts.append(f"Emotion: {em}")
            if parts:
                label = " | ".join(parts)
                cx_text = (x1 + x2) // 2
                (tw, th), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.FONT_SCALE,
                    config.FONT_THICKNESS,
                )
                tx = max(cx_text - tw // 2, 4)
                ty = max(y1 - 8, th + 4)
                cv2.putText(
                    frame,
                    label,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.FONT_SCALE,
                    (0, 255, 255),
                    config.FONT_THICKNESS,
                    lineType=cv2.LINE_AA,
                )

        if show_labels and "spoof_real" in face and "spoof_confidence" in face:
            real = face["spoof_real"]
            conf = face["spoof_confidence"]
            spoof_label = f"Real Human: {int(conf * 100) if real else 0}%"
            cx_s = (x1 + x2) // 2
            (sw, sh), _ = cv2.getTextSize(
                spoof_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                config.FONT_THICKNESS,
            )
            sx = max(cx_s - sw // 2, 4)
            sy = min(y2 + sh + 8, frame.shape[0] - 4)
            cv2.putText(
                frame,
                spoof_label,
                (sx, sy),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                (0, 255, 255),
                config.FONT_THICKNESS,
                lineType=cv2.LINE_AA,
            )

    return frame


def apply_privacy(
    frame: np.ndarray,
    faces,
    mode: str,
    inplace: bool = False,
) -> np.ndarray:
    """Blur face regions in *frame* using the selected *mode*.

    Modes: "Pixelate", "Gaussian", "EllipticalBlur".  Pass "None" to return
    the frame unchanged.
    """
    if mode == "None" or not faces:
        return frame if inplace else frame.copy()

    bboxes = [f["bbox"] for f in faces]
    try:
        from uniface.privacy import BlurFace

        method = mode.lower()
        if method == "ellipticalblur":
            method = "elliptical"
        bf = BlurFace(method=method)
        return bf.blur_regions(frame, bboxes, inplace)
    except Exception:
        return frame if inplace else frame.copy()


# ---------------------------------------------------------------------------
# FaceEngine — lazy-loaded MediaPipe Face Landmarker
# ---------------------------------------------------------------------------


class FaceEngine:
    """Lazy-loaded MediaPipe Face Landmarker + UniFace attribute analysis.

    Uses CPU delegate — the GPU (Metal) delegate crashes on macOS with
    ``kCVReturnPixelBufferNotMetalCompatible`` even with 4-channel input,
    and is unreliable beyond a few minutes.
    UniFace models (headpose, age/gender, emotion) are lazy-loaded on
    first face detection.
    """

    def __init__(self):
        self._landmarker = None
        self._lock = threading.Lock()
        self._frame_count = 0
        self._headpose = None
        self._age_gender = None
        self._emotion = None
        self._spoofing = None
        self._background_remover = None
        self._bg_alpha_cache = None
        self._bg_frame_counter = 0
        self._uniface_lock = threading.Lock()
        self._mediapipe_5pt = None

    def ensure_loaded(self):
        if self._landmarker is not None:
            return
        with self._lock:
            if self._landmarker is not None:
                return

            model_path = download_model()

            import mediapipe as mp  # noqa: F401 — needed for mp.Image below
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_opts = python.BaseOptions(
                model_asset_path=str(model_path),
                delegate=python.BaseOptions.Delegate.CPU,
            )
            opts = vision.FaceLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.VIDEO,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=2,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(opts)
            logger.info("Face Landmarker loaded (CPU)")

    def _ensure_uniface(self):
        """Lazy-load UniFace attribute models."""
        if self._headpose is not None:
            return
        with self._uniface_lock:
            if self._headpose is not None:
                return

            from uniface.attribute import AgeGender
            from uniface.attribute.emotion import Emotion
            from uniface.headpose import HeadPose
            from uniface.spoofing import MiniFASNet

            logger.info("Loading UniFace attribute models...")
            self._headpose = HeadPose()  # type: ignore[no-untyped-call]
            self._age_gender = AgeGender()  # type: ignore[no-untyped-call]
            self._emotion = Emotion()  # type: ignore[no-untyped-call]
            self._spoofing = MiniFASNet()  # type: ignore[no-untyped-call]
            try:
                from uniface.matting import MODNet, MODNetWeights

                self._background_remover = MODNet(
                    model_name=MODNetWeights.WEBCAM,
                    input_size=192,
                )  # type: ignore[no-untyped-call]
                self._bg_alpha_cache = None
                logger.info("Background remover loaded (webcam, 192px)")
            except Exception:
                logger.warning("Background remover unavailable")
            logger.info("UniFace models ready")

    def _estimate_headpose(self, face_crop):
        try:
            from uniface.types import HeadPoseResult

            hp = self._headpose.estimate(face_crop)  # pyright: ignore
            if isinstance(hp, HeadPoseResult):
                return {
                    "pitch": float(hp.pitch),
                    "yaw": float(hp.yaw),
                    "roll": float(hp.roll),
                }
        except Exception:
            pass
        return None

    def _predict_attributes(self, frame, bbox, pts5):
        """Return (age, gender_str, emotion_str) or None-filled tuple."""
        age = None
        gender = None
        emotion = None

        try:
            from uniface.types import AttributeResult
            from uniface.types import Face as UniFace

            uf_face = UniFace(
                bbox=np.array(bbox, dtype=np.float64),
                confidence=0.95,
                landmarks=np.array(pts5, dtype=np.float64).reshape(-1, 2),
            )
            ag = self._age_gender.predict(frame, uf_face)  # pyright: ignore
            if isinstance(ag, AttributeResult):
                gender = "Male" if ag.gender == 1 else "Female"
                age = ag.age if ag.age is not None else 0
        except Exception:
            pass

        try:
            from uniface.types import EmotionResult
            from uniface.types import Face as UniFace2

            uf_face2 = UniFace2(
                bbox=np.array(bbox, dtype=np.float64),
                confidence=0.95,
                landmarks=np.array(pts5, dtype=np.float64).reshape(-1, 2),
            )
            em = self._emotion.predict(frame, uf_face2)  # pyright: ignore
            if isinstance(em, EmotionResult):
                emotion = str(em.emotion)
        except Exception:
            pass

        return age, gender, emotion

    def remove_background(self, frame: np.ndarray) -> np.ndarray:
        """Replace background with white using MODNet alpha matting.

        Runs inference on a downscaled frame (2x smaller) for speed
        and caches the alpha matte for 5 frames.
        """
        if self._background_remover is None:
            return frame
        try:
            self._bg_frame_counter += 1
            if self._bg_alpha_cache is not None and self._bg_frame_counter % 2 != 0:
                alpha = self._bg_alpha_cache
            else:
                h, w = frame.shape[:2]
                small = cv2.resize(
                    frame, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR
                )
                alpha_small = self._background_remover.predict(small)  # pyright: ignore
                self._bg_alpha_cache = cv2.resize(
                    alpha_small,
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                alpha = self._bg_alpha_cache
            return (frame * alpha[..., None] + 255 * (1 - alpha[..., None])).astype(
                np.uint8
            )
        except Exception:
            return frame

    def process(self, frame: np.ndarray):
        """Run face landmark detection on *frame*.

        Returns a list of dicts, one per detected face::

            {
                "label": "Face",
                "confidence": 0.95,
                "bbox": (x1, y1, x2, y2),
                "landmarks": [(x, y), ...],
                "headpose": {"pitch": ..., "yaw": ..., "roll": ...},
                "age": 32,
                "gender": "Male",
                "emotion": "Happy",
            }

        Returns an empty list when no face is detected.
        """
        self.ensure_loaded()
        assert self._landmarker is not None

        self._frame_count += 1
        timestamp_ms = self._frame_count * 33  # ~30 FPS pacing

        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        faces = []
        if not result.face_landmarks:
            return faces

        self._ensure_uniface()

        h, w = frame.shape[:2]
        for landmarks in result.face_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1 = max(0, min(xs)), max(0, min(ys))
            x2, y2 = min(w - 1, max(xs)), min(h - 1, max(ys))

            face_dict = {
                "label": "Face",
                "confidence": 0.95,
                "bbox": (x1, y1, x2, y2),
                "landmarks": pts,
            }

            if self._headpose is not None and x2 > x1 and y2 > y1:
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    hp = self._estimate_headpose(face_crop)
                    if hp is not None:
                        face_dict["headpose"] = hp  # pyright: ignore

            if (
                self._age_gender is not None
                and self._emotion is not None
                and x2 > x1
                and y2 > y1
            ):
                self._mediapipe_5pt = [
                    (
                        sum(pts[i][0] for i in LEFT_EYE)
                        // len(LEFT_EYE),  # left eye center
                        sum(pts[i][1] for i in LEFT_EYE) // len(LEFT_EYE),
                    ),
                    (
                        sum(pts[i][0] for i in RIGHT_EYE)
                        // len(RIGHT_EYE),  # right eye center
                        sum(pts[i][1] for i in RIGHT_EYE) // len(RIGHT_EYE),
                    ),
                    pts[1],  # nose tip
                    pts[61],  # left mouth corner
                    pts[291],  # right mouth corner
                ]
                age, gender, emotion = self._predict_attributes(
                    frame, (x1, y1, x2, y2), self._mediapipe_5pt
                )
                if age is not None:
                    face_dict["age"] = age
                if gender is not None:
                    face_dict["gender"] = gender
                if emotion is not None:
                    face_dict["emotion"] = emotion

            if self._spoofing is not None and x2 > x1 and y2 > y1:
                try:
                    sr = self._spoofing.predict(frame, [x1, y1, x2, y2])  # pyright: ignore
                    face_dict["spoof_real"] = sr.is_real
                    face_dict["spoof_confidence"] = float(sr.confidence)
                except Exception:
                    pass

            faces.append(face_dict)

        return faces
