"""MediaPipe Face Landmarker — 478-point 3D face mesh for Apple Silicon."""

from __future__ import annotations

import json
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
from uniface.attribute import AgeGender
from uniface.attribute.emotion import Emotion
from uniface.constants import MobileFaceWeights, SCRFDWeights
from uniface.detection import SCRFD
from uniface.face_utils import compute_similarity
from uniface.headpose import HeadPose
from uniface.privacy import BlurFace
from uniface.recognition import MobileFace
from uniface.spoofing import MiniFASNet
from uniface.tracking import BYTETracker
from uniface.types import AttributeResult, EmotionResult, HeadPoseResult
from uniface.types import Face as UniFace

from src import config

logger = logging.getLogger(__name__)

MODEL_DIR = Path(config.MODELS_DIR)
MODEL_FILE = MODEL_DIR / "face_landmarker_v2.task"
FACE_IDENTITIES_PATH = MODEL_DIR / "identities.json"
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
    overlay_color=config.OVERLAY_COLOR,
    font_scale: float = config.FONT_SCALE,
    font_thickness: int = config.FONT_THICKNESS,
    line_thickness: int = config.OVERLAY_THICKNESS,
    face_id_names: dict[int, str] | None = None,
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
    face_id_names : dict[int, str] | None
        Optional mapping of track_id to user-assigned name.
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
                    color=overlay_color,
                    thickness=line_thickness,
                )

            for i in range(0, len(pts), config.FACE_POINT_STRIDE):
                cv2.circle(
                    frame,
                    (pts[i][0], pts[i][1]),
                    2,
                    overlay_color,
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
                parts.append(f"Gender: {face['gender'][0].upper()}")
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
                    font_scale,
                    font_thickness,
                )
                tx = max(cx_text - tw // 2, 4)
                ty = max(y1 - 8, th + 4)
                cv2.putText(
                    frame,
                    label,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 255),
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )

        if show_labels and "spoof_confidence" in face:
            conf = face["spoof_confidence"]
            spoof_label = f"Human: {int(conf * 100)}%"
            if "track_id" in face:
                tid = face["track_id"]
                name = face_id_names.get(tid, "") if face_id_names else ""
                if name:
                    spoof_label = f"{name.upper()} | " + spoof_label
                else:
                    spoof_label = f"ID: {tid} | " + spoof_label

            cx_s = (x1 + x2) // 2
            (sw, sh), _ = cv2.getTextSize(
                spoof_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
            )
            sx = max(cx_s - sw // 2, 4)
            sy = min(y2 + sh + 8, frame.shape[0] - 4)
            cv2.putText(
                frame,
                spoof_label,
                (sx, sy),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 255),
                font_thickness,
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
        self._uniface_lock = threading.Lock()
        self._mediapipe_5pt = None
        self._face_id_detector = None
        self._bytetracker = None
        self._face_recognizer = None
        self._known_face_embeddings: dict[str, list[float]] = {}
        self._track_identity_cache: dict[int, str] = {}

    def ensure_loaded(self):
        if self._landmarker is not None:
            return
        with self._lock:
            if self._landmarker is not None:
                return

            model_path = download_model()

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

    def warmup(self):
        self.ensure_loaded()
        self._ensure_uniface(
            need_headpose=True,
            need_labels=True,
        )

        dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)
        dummy_bbox = (64, 64, 192, 192)
        dummy_points = [
            (96, 112),
            (160, 112),
            (128, 144),
            (104, 176),
            (152, 176),
        ]

        try:
            self.process(dummy_frame, show_headpose=False, show_labels=False)
        except Exception as exc:
            logger.warning("Face landmarker warmup inference failed: %s", exc)

        try:
            self._estimate_headpose(dummy_frame[64:192, 64:192])
        except Exception as exc:
            logger.warning("Headpose warmup inference failed: %s", exc)

        try:
            self._predict_attributes(dummy_frame, dummy_bbox, dummy_points)
        except Exception as exc:
            logger.warning("Attribute warmup inference failed: %s", exc)

        if self._spoofing is not None:
            try:
                self._spoofing.predict(dummy_frame, list(dummy_bbox))  # pyright: ignore
            except Exception as exc:
                logger.warning("Spoofing warmup inference failed: %s", exc)

    def _ensure_uniface(
        self,
        need_headpose: bool = False,
        need_labels: bool = False,
    ):
        """Lazy-load only the UniFace models needed by enabled features."""
        if need_headpose and self._headpose is None:
            with self._uniface_lock:
                if self._headpose is None:
                    logger.info("Loading UniFace headpose model...")
                    self._headpose = HeadPose()  # type: ignore[no-untyped-call]

        if need_labels and self._age_gender is None:
            with self._uniface_lock:
                if self._age_gender is None:
                    logger.info("Loading UniFace label models...")
                    self._age_gender = AgeGender()  # type: ignore[no-untyped-call]
                    self._emotion = Emotion()  # type: ignore[no-untyped-call]
                    self._spoofing = MiniFASNet()  # type: ignore[no-untyped-call]

    def _estimate_headpose(self, face_crop):
        try:
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
            uf_face2 = UniFace(
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

    def _load_known_face_embeddings(self):
        if self._known_face_embeddings:
            return
        try:
            if FACE_IDENTITIES_PATH.exists():
                with FACE_IDENTITIES_PATH.open() as file_handle:
                    data = json.load(file_handle)
                self._known_face_embeddings = {
                    str(name): [float(value) for value in embedding]
                    for name, embedding in data.items()
                    if isinstance(embedding, list)
                }
        except Exception as exc:
            logger.warning("Failed to load face identities: %s", exc)

    def _save_known_face_embeddings(self):
        try:
            with FACE_IDENTITIES_PATH.open("w") as file_handle:
                json.dump(self._known_face_embeddings, file_handle, indent=2)
        except Exception as exc:
            logger.warning("Failed to save face identities: %s", exc)

    def _ensure_face_id_models(self, load_recognizer: bool = False):
        """Lazy-load SCRFD/ByteTrack, and MobileFace only when needed."""
        if self._face_id_detector is None:
            with self._uniface_lock:
                if self._face_id_detector is None:
                    logger.info("Loading UniFace face ID detector (SCRFD 500M)...")
                    self._face_id_detector = SCRFD(
                        model_name=SCRFDWeights.SCRFD_500M_KPS,
                    )
                    self._bytetracker = BYTETracker()
        if load_recognizer and self._face_recognizer is None:
            with self._uniface_lock:
                if self._face_recognizer is None:
                    logger.info("Loading UniFace MobileFace recognizer (MNET_025)...")
                    self._face_recognizer = MobileFace(
                        model_name=MobileFaceWeights.MNET_025,
                    )
                    self._load_known_face_embeddings()

    @staticmethod
    def _bbox_iou(bbox_a, bbox_b) -> float:
        """Compute IoU between two bboxes in (x1, y1, x2, y2) format."""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        xi1 = max(ax1, bx1)
        yi1 = max(ay1, by1)
        xi2 = min(ax2, bx2)
        yi2 = min(ay2, by2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / (area_a + area_b - inter + 1e-6)

    @staticmethod
    def _get_five_point_landmarks(landmarks: list[tuple[int, int]]) -> np.ndarray:
        return np.array(
            [
                (
                    sum(landmarks[index][0] for index in LEFT_EYE) // len(LEFT_EYE),
                    sum(landmarks[index][1] for index in LEFT_EYE) // len(LEFT_EYE),
                ),
                (
                    sum(landmarks[index][0] for index in RIGHT_EYE) // len(RIGHT_EYE),
                    sum(landmarks[index][1] for index in RIGHT_EYE) // len(RIGHT_EYE),
                ),
                landmarks[1],
                landmarks[61],
                landmarks[291],
            ],
            dtype=np.float32,
        )

    def _recognize_detection(self, frame: np.ndarray, detection: dict) -> str | None:
        if not self._known_face_embeddings:
            return None
        self._ensure_face_id_models(load_recognizer=True)
        assert self._face_recognizer is not None
        try:
            embedding = self._face_recognizer.get_normalized_embedding(
                frame,
                self._get_five_point_landmarks(detection["landmarks"]),
            )
        except Exception as exc:
            logger.warning("Face recognition failed: %s", exc)
            return None

        best_name = None
        best_similarity = config.FACE_ID_SIMILARITY_THRESHOLD
        for name, stored_embedding in self._known_face_embeddings.items():
            similarity = float(
                compute_similarity(
                    np.array(stored_embedding, dtype=np.float32),
                    embedding,
                    normalized=True,
                )
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name
        return best_name

    def enroll_identity(
        self, name: str, frame: np.ndarray, detection: dict, track_id: int
    ):
        self._ensure_face_id_models(load_recognizer=True)
        assert self._face_recognizer is not None
        embedding = self._face_recognizer.get_normalized_embedding(
            frame,
            self._get_five_point_landmarks(detection["landmarks"]),
        )
        self._known_face_embeddings[name] = embedding.astype(float).tolist()
        self._track_identity_cache[track_id] = name
        self._save_known_face_embeddings()

    def remove_identity(self, name: str, track_id: int | None = None):
        self._known_face_embeddings.pop(name, None)
        if track_id is not None:
            self._track_identity_cache.pop(track_id, None)
        else:
            self._track_identity_cache = {
                cached_track_id: cached_name
                for cached_track_id, cached_name in self._track_identity_cache.items()
                if cached_name != name
            }
        self._save_known_face_embeddings()

    def apply_face_identities(
        self, frame: np.ndarray, detections: list[dict]
    ) -> list[dict]:
        """Track every frame, recognize only newly seen track IDs."""
        self.track_face_ids(frame, detections)

        active_track_ids = {
            int(detection["track_id"])
            for detection in detections
            if "track_id" in detection
        }
        self._track_identity_cache = {
            track_id: name
            for track_id, name in self._track_identity_cache.items()
            if track_id in active_track_ids
        }

        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            cached_name = self._track_identity_cache.get(int(track_id))
            if cached_name is not None:
                detection["identity_name"] = cached_name
                continue
            identity_name = self._recognize_detection(frame, detection)
            if identity_name is not None:
                self._track_identity_cache[int(track_id)] = identity_name
                detection["identity_name"] = identity_name

        return detections

    def merge_face_identity_results(
        self,
        detections: list[dict],
        identity_results: list[dict],
    ) -> list[dict]:
        for detection in detections:
            best_match = None
            best_iou = 0.0
            for identity_result in identity_results:
                iou = self._bbox_iou(detection["bbox"], identity_result["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_match = identity_result
            if best_match is None or best_iou <= 0.3:
                continue
            if "track_id" in best_match:
                detection["track_id"] = int(best_match["track_id"])
            if "identity_name" in best_match:
                detection["identity_name"] = str(best_match["identity_name"])
        return detections

    def track_face_ids(self, frame: np.ndarray, detections: list[dict]) -> list[dict]:
        """Attach persistent track IDs to face dicts via detector + ByteTrack.

        Runs SCRFD face detection on the full frame, feeds detections into
        ByteTrack, then associates track IDs with the existing MediaPipe face
        dicts by IoU matching.  Returns the same ``detections`` list with
        ``track_id`` set in-place.
        """
        self._ensure_face_id_models()
        assert self._face_id_detector is not None
        assert self._bytetracker is not None

        # Run face detector
        uniface_faces = self._face_id_detector.detect(frame)
        if not uniface_faces:
            return detections

        # Build (N, 5) array for ByteTrack: [x1, y1, x2, y2, score]
        dets = np.array(
            [
                [f.bbox[0], f.bbox[1], f.bbox[2], f.bbox[3], f.confidence]
                for f in uniface_faces
            ],
            dtype=np.float32,
        )

        # Update tracker -> (M, 5) with [x1, y1, x2, y2, track_id]
        tracks = self._bytetracker.update(dets)
        if len(tracks) == 0:
            return detections

        # Associate track IDs with MediaPipe faces via best IoU
        for det_face in detections:
            db = det_face["bbox"]
            best_iou = 0.0
            best_id = None
            for track in tracks:
                iou = self._bbox_iou(db, (track[0], track[1], track[2], track[3]))
                if iou > best_iou:
                    best_iou = iou
                    best_id = int(track[4])
            if best_id is not None and best_iou > 0.3:
                det_face["track_id"] = best_id

        return detections

    def process(
        self,
        frame: np.ndarray,
        show_headpose: bool = True,
        show_labels: bool = True,
    ):
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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_img, timestamp_ms)

        faces = []
        if not result.face_landmarks:
            return faces

        self._ensure_uniface(
            need_headpose=show_headpose,
            need_labels=show_labels,
        )

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

            if show_headpose and self._headpose is not None and x2 > x1 and y2 > y1:
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    hp = self._estimate_headpose(face_crop)
                    if hp is not None:
                        face_dict["headpose"] = hp  # pyright: ignore

            if (
                show_labels
                and self._age_gender is not None
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

            if show_labels and self._spoofing is not None and x2 > x1 and y2 > y1:
                try:
                    sr = self._spoofing.predict(frame, [x1, y1, x2, y2])  # pyright: ignore
                    face_dict["spoof_real"] = sr.is_real
                    face_dict["spoof_confidence"] = float(sr.confidence)
                except Exception:
                    pass

            faces.append(face_dict)

        return faces
