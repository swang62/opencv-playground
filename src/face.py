"""MediaPipe Face Landmarker — 478-point 3D face mesh for Apple Silicon."""

from __future__ import annotations

import concurrent.futures
import logging
import ssl
import threading
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from uniface.attribute import AgeGender, FairFace
from uniface.attribute.emotion import Emotion
from uniface.constants import ArcFaceWeights, SCRFDWeights
from uniface.detection import SCRFD
from uniface.privacy import BlurFace
from uniface.recognition import ArcFace
from uniface.spoofing import MiniFASNet
from uniface.stores import FAISS
from uniface.tracking import BYTETracker
from uniface.types import AttributeResult, EmotionResult
from uniface.types import Face as UniFace

from src import config

logger = logging.getLogger(__name__)

MODEL_DIR = Path(config.MODELS_DIR)
MODEL_FILE = MODEL_DIR / "face_landmarker_v2.task"
FACE_IDENTITIES_PATH = MODEL_DIR / "identities"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# ---------------------------------------------------------------------------
# Face mesh connections — (start, end) index pairs per feature from the
# MediaPipe Face Mesh canonical 478-point topology (FaceLandmarksConnections).
# Using connection pairs (not ordered point lists) because several features
# have disconnected segments that polylines would bridge incorrectly.
# ---------------------------------------------------------------------------

FACE_OVAL = [
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10),
]

LEFT_EYE = [
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
]

RIGHT_EYE = [
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
]

LEFT_BROW = [
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
]

RIGHT_BROW = [
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
]

NOSE = [
    (168, 6),
    (6, 197),
    (197, 195),
    (195, 5),
    (5, 4),
    (4, 1),
    (1, 19),
    (19, 94),
    (94, 2),
]

LIPS = [
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308),
]

ALL_CONTOURS = [
    FACE_OVAL,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_BROW,
    RIGHT_BROW,
    NOSE,
    LIPS,
]

# Unique landmark indices used for eye-center averaging in 5-point landmark
# computation (not for drawing).
LEFT_EYE_INDICES = sorted({i for pair in LEFT_EYE for i in pair})
RIGHT_EYE_INDICES = sorted({i for pair in RIGHT_EYE for i in pair})


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


def draw_face_mesh(
    frame: np.ndarray,
    faces,
    show_wireframe: bool = True,
    show_labels: bool = True,
    overlay_color=config.OVERLAY_COLOR,
    font_scale: float = config.FONT_SCALE,
    font_thickness: int = config.FONT_THICKNESS,
    line_thickness: int = config.OVERLAY_THICKNESS,
    face_id_names: dict[int, str] | None = None,
) -> np.ndarray:
    """Draw face mesh and attribute labels on *frame*.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame (modified in-place for speed).
    faces : list[dict]
        Each dict has ``"landmarks"`` (478 (x, y) tuples), ``"bbox"``, and
        optional ``"age"``, ``"gender"``, ``"emotion"`` keys.
    fps : float
        Current frames-per-second to overlay.
    show_wireframe : bool
        Whether to draw the 478-point mesh wireframe.
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
            for conn_group in ALL_CONTOURS:
                for i, j in conn_group:
                    if i < len(pts) and j < len(pts):
                        cv2.line(
                            frame,
                            pts[i],
                            pts[j],
                            overlay_color,
                            line_thickness,
                            cv2.LINE_AA,
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

        if show_labels:
            parts = []
            if "age" in face:
                parts.append(f"Age: {face['age']}")
            if "gender" in face:
                parts.append(f"{face['gender']}")
            if "race" in face:
                parts.append(f"{face['race']}")
            if "emotion" in face:
                em = face["emotion"]
                parts.append(f"{em}")
            if parts:
                label = " | ".join(parts)
                cx_text = (x1 + x2) // 2
                (tw, th), _ = cv2.getTextSize(
                    label,
                    config.OVERLAY_FONT,
                    font_scale,
                    font_thickness,
                )
                tx = max(cx_text - tw // 2, 4)
                ty = max(y1 - 8, th + 4)
                cv2.putText(
                    frame,
                    label,
                    (tx, ty),
                    config.OVERLAY_FONT,
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
                    spoof_label = f"{name} | " + spoof_label
                else:
                    spoof_label = f"ID: {tid} | " + spoof_label

            cx_s = (x1 + x2) // 2
            (sw, sh), _ = cv2.getTextSize(
                spoof_label,
                config.OVERLAY_FONT,
                font_scale,
                font_thickness,
            )
            sx = max(cx_s - sw // 2, 4)
            sy = min(y2 + sh + 8, frame.shape[0] - 4)
            cv2.putText(
                frame,
                spoof_label,
                (sx, sy),
                config.OVERLAY_FONT,
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
    UniFace models (age/gender, emotion) are lazy-loaded on
    first face detection.
    """

    def __init__(self):
        self._landmarker = None
        self._lock = threading.Lock()
        self._frame_count = 0
        self._age_gender = None
        self._emotion = None
        self._spoofing = None
        self._race = None
        self._uniface_lock = threading.Lock()
        self._mediapipe_5pt = None
        self._face_id_detector = None
        self._bytetracker = None
        self._face_recognizer = None
        self._face_store = None
        self._track_identity_cache: dict[int, dict[str, str]] = {}
        self._track_embedding_buffers: dict[int, deque] = {}
        self._smoothed_ages: list[
            tuple
        ] = []  # [(bbox, smoothed_age), ...] for EMA across frames
        self._label_cache: list[
            dict
        ] = []  # recent label results keyed by bbox for IoU-stable replay
        self._race_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="fairface"
        )
        self._race_future: concurrent.futures.Future | None = None
        self._race_results: list[tuple[tuple, str]] = []  # [(bbox, race_str), ...]

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
        self._ensure_uniface(need_labels=True)

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
            self.process(dummy_frame, show_labels=False)
        except Exception as exc:
            logger.warning("Face landmarker warmup inference failed: %s", exc)

        try:
            self._predict_attributes(dummy_frame, dummy_bbox, dummy_points)
        except Exception as exc:
            logger.warning("Attribute warmup inference failed: %s", exc)

        if self._spoofing is not None:
            try:
                self._spoofing.predict(dummy_frame, list(dummy_bbox))  # pyright: ignore
            except Exception as exc:
                logger.warning("Spoofing warmup inference failed: %s", exc)

    def _ensure_uniface(self, need_labels: bool = False):
        """Lazy-load only the UniFace models needed by enabled features."""
        if need_labels and self._age_gender is None:
            with self._uniface_lock:
                if self._age_gender is None:
                    logger.info("Loading UniFace label models...")
                    onnx_providers = [
                        "CoreMLExecutionProvider",
                        "CPUExecutionProvider",
                    ]
                    self._age_gender = AgeGender(  # type: ignore[no-untyped-call]
                        providers=onnx_providers,
                    )
                    self._emotion = Emotion()  # type: ignore[no-untyped-call]
                    self._emotion.device = torch.device("cpu")
                    self._emotion.model = self._emotion.model.to("cpu")
                    self._spoofing = MiniFASNet(  # type: ignore[no-untyped-call]
                        providers=onnx_providers,
                    )
                    self._race = FairFace(  # type: ignore[no-untyped-call]
                        providers=onnx_providers,
                    )
                    logger.info(
                        "UniFace label models loaded (age/gender, emotion, spoofing, race)"
                    )

    def _predict_attributes(self, frame, bbox, pts5):
        """Return (age, gender, emotion) or None-filled tuple."""
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

    def _ensure_face_store(self):
        if self._face_store is not None:
            return self._face_store
        with self._uniface_lock:
            if self._face_store is not None:
                return self._face_store
            FACE_IDENTITIES_PATH.mkdir(parents=True, exist_ok=True)
            store = FAISS(db_path=str(FACE_IDENTITIES_PATH))
            loaded = store.load()
            logger.info(
                "Face identity store ready: path=%s loaded=%s size=%s",
                FACE_IDENTITIES_PATH,
                loaded,
                len(store),
            )
            if not loaded:
                logger.info(
                    "Face identity store is empty; save a face to create entries"
                )
            self._face_store = store
            return self._face_store

    def _ensure_face_id_models(self, load_recognizer: bool = False):
        """Lazy-load SCRFD/ByteTrack, and MobileFace only when needed."""
        onnx_providers = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        if self._face_id_detector is None:
            with self._uniface_lock:
                if self._face_id_detector is None:
                    logger.info("Loading UniFace face ID detector (SCRFD 500M)...")
                    self._face_id_detector = SCRFD(
                        model_name=SCRFDWeights.SCRFD_500M_KPS,
                        input_size=config.FACE_DETECTION_INPUT_SIZE,
                        providers=onnx_providers,
                    )
                    self._bytetracker = BYTETracker()
        if load_recognizer and self._face_recognizer is None:
            with self._uniface_lock:
                if self._face_recognizer is None:
                    logger.info("Loading UniFace ArcFace recognizer (RESNET)...")
                    self._face_recognizer = ArcFace(
                        model_name=ArcFaceWeights.RESNET,
                        providers=onnx_providers,
                    )
        if load_recognizer:
            self._ensure_face_store()

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

    def _smooth_age(self, bbox: tuple, raw_age: float) -> int:
        best_iou = 0.0
        best_age = None
        best_idx = -1
        for idx, (prev_bbox, prev_age) in enumerate(self._smoothed_ages):
            iou = self._bbox_iou(bbox, prev_bbox)
            if iou > best_iou:
                best_iou = iou
                best_age = prev_age
                best_idx = idx
        if best_iou > 0.3 and best_age is not None:
            smoothed = (
                1 - config.AGE_SMOOTHING_ALPHA
            ) * best_age + config.AGE_SMOOTHING_ALPHA * raw_age
        else:
            smoothed = float(raw_age)
        new_entry = (bbox, smoothed)
        if best_idx >= 0:
            self._smoothed_ages[best_idx] = new_entry
        else:
            self._smoothed_ages.append(new_entry)
            self._smoothed_ages = self._smoothed_ages[-4:]
        return round(smoothed)

    def _match_label_cache(self, bbox) -> dict | None:
        best_iou = 0.0
        best_entry: dict | None = None
        for entry in self._label_cache:
            iou = self._bbox_iou(bbox, entry["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_entry = entry
        # Always return the best match (no hard threshold) — even a slightly stale cached
        # label with a displaced bbox is less visually jarring than labels flickering off.
        return best_entry

    @staticmethod
    def _get_five_point_landmarks(landmarks: list[tuple[int, int]]) -> np.ndarray:
        return np.array(
            [
                (
                    sum(landmarks[i][0] for i in LEFT_EYE_INDICES)
                    // len(LEFT_EYE_INDICES),
                    sum(landmarks[i][1] for i in LEFT_EYE_INDICES)
                    // len(LEFT_EYE_INDICES),
                ),
                (
                    sum(landmarks[i][0] for i in RIGHT_EYE_INDICES)
                    // len(RIGHT_EYE_INDICES),
                    sum(landmarks[i][1] for i in RIGHT_EYE_INDICES)
                    // len(RIGHT_EYE_INDICES),
                ),
                landmarks[1],
                landmarks[61],
                landmarks[291],
            ],
            dtype=np.float32,
        )

    def _recognize_detection(self, frame: np.ndarray, detection: dict, track_id: int):
        try:
            self._ensure_face_id_models(load_recognizer=True)
            store = self._ensure_face_store()
        except Exception as exc:
            logger.warning("Face recognition store unavailable: %s", exc)
            return None
        if len(store) == 0:
            return None
        assert self._face_recognizer is not None
        try:
            embedding = self._face_recognizer.get_normalized_embedding(
                frame,
                self._get_five_point_landmarks(detection["landmarks"]),
            )
        except Exception as exc:
            logger.warning("Face recognition failed: %s", exc)
            return None

        buf = self._track_embedding_buffers.setdefault(
            track_id, deque(maxlen=config.REID_EMBEDDING_BUFFER_SIZE)
        )
        buf.append(embedding)
        if len(buf) < config.REID_EMBEDDING_BUFFER_SIZE:
            return None

        mean_emb = np.mean(list(buf), axis=0)

        logger.debug(
            "Face recognition scanning %d saved identities (buffered=%d)",
            len(store),
            len(buf),
        )
        result, similarity = store.search(
            mean_emb,
            threshold=config.SIMILARITY_THRESHOLD,
        )
        logger.debug(
            "Face similarity search: result=%s score=%.4f threshold=%.4f",
            result,
            similarity,
            config.SIMILARITY_THRESHOLD,
        )
        if result is None:
            logger.debug("Face recognition found no match above threshold")
            return None
        matched_name = str(result.get("name", ""))
        logger.info(
            "Face recognition matched name=%s score=%.4f",
            matched_name,
            similarity,
        )
        return {"identity_name": matched_name}

    def enroll_identity(
        self, name: str, frame: np.ndarray, detection: dict, track_id: int
    ):
        self._ensure_face_id_models(load_recognizer=True)
        store = self._ensure_face_store()
        assert self._face_recognizer is not None
        embedding = self._face_recognizer.get_normalized_embedding(
            frame,
            self._get_five_point_landmarks(detection["landmarks"]),
        )
        buf = self._track_embedding_buffers.get(track_id)
        if buf:
            all_embs = list(buf) + [embedding]
            embedding = np.mean(all_embs, axis=0)
        store.remove("name", name)
        store.add(embedding, {"name": name})
        store.save()
        self._track_identity_cache[track_id] = {"identity_name": name}
        logger.info(
            "Saved face identity: name=%s track=%s total_saved=%d",
            name,
            track_id,
            len(store),
        )

    def remove_identity(self, name: str, track_id: int | None = None):
        store = self._ensure_face_store()
        removed = store.remove("name", name)
        if removed:
            store.save()
        if track_id is not None:
            self._track_identity_cache.pop(track_id, None)
        else:
            self._track_identity_cache = {
                cached_track_id: cached_identity
                for cached_track_id, cached_identity in self._track_identity_cache.items()
                if cached_identity.get("identity_name") != name
            }

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
            track_id: identity
            for track_id, identity in self._track_identity_cache.items()
            if track_id in active_track_ids
        }
        stale_buffers = set(self._track_embedding_buffers) - active_track_ids
        for tid in stale_buffers:
            self._track_embedding_buffers.pop(tid, None)

        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            tid = int(track_id)
            cached_identity = self._track_identity_cache.get(tid)
            if cached_identity is not None:
                detection["identity_name"] = cached_identity["identity_name"]
                continue
            try:
                matched_identity = self._recognize_detection(frame, detection, tid)
            except Exception as exc:
                logger.warning(
                    "Face recognition lookup failed for track=%s: %s", track_id, exc
                )
                matched_identity = None
            if matched_identity is not None:
                self._track_identity_cache[int(track_id)] = matched_identity
                if matched_identity["identity_name"]:
                    detection["identity_name"] = matched_identity["identity_name"]

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

    def _background_race_inference(self, frame, face_data):
        """Run FairFace on all faces in a background thread.

        Returns list of (bbox_tuple, race_str) for faces where race was predicted.
        """
        if self._race is None:
            return []
        results = []
        for bbox, pts5 in face_data:
            try:
                uf = UniFace(
                    bbox=np.array(bbox, dtype=np.float64),
                    confidence=0.95,
                    landmarks=np.array(pts5, dtype=np.float64).reshape(-1, 2),
                )
                rr = self._race.predict(frame, uf)  # pyright: ignore
                if isinstance(rr, AttributeResult) and rr.race:
                    results.append((bbox, rr.race))
            except Exception:
                pass
        return results

    def process(
        self,
        frame: np.ndarray,
        show_labels: bool = True,
    ):
        """Run face landmark detection on *frame*.

        Returns a list of dicts, one per detected face::

            {
                "label": "Face",
                "confidence": 0.95,
                "bbox": (x1, y1, x2, y2),
                "landmarks": [(x, y), ...],
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

        self._ensure_uniface(need_labels=show_labels)

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

            if show_labels and x2 > x1 and y2 > y1:
                if (
                    self._frame_count % config.INFERENCE_UPDATE_INTERVAL == 0
                    and self._age_gender is not None
                    and self._emotion is not None
                    and self._spoofing is not None
                ):
                    self._mediapipe_5pt = [
                        (
                            sum(pts[i][0] for i in LEFT_EYE_INDICES)
                            // len(LEFT_EYE_INDICES),  # left eye center
                            sum(pts[i][1] for i in LEFT_EYE_INDICES)
                            // len(LEFT_EYE_INDICES),
                        ),
                        (
                            sum(pts[i][0] for i in RIGHT_EYE_INDICES)
                            // len(RIGHT_EYE_INDICES),  # right eye center
                            sum(pts[i][1] for i in RIGHT_EYE_INDICES)
                            // len(RIGHT_EYE_INDICES),
                        ),
                        pts[1],  # nose tip
                        pts[61],  # left mouth corner
                        pts[291],  # right mouth corner
                    ]
                    age, gender, emotion = self._predict_attributes(
                        frame, (x1, y1, x2, y2), self._mediapipe_5pt
                    )
                    if age is not None:
                        face_dict["age"] = self._smooth_age((x1, y1, x2, y2), age)
                    if gender is not None:
                        face_dict["gender"] = gender
                    if emotion is not None:
                        face_dict["emotion"] = emotion

                    try:
                        sr = self._spoofing.predict(frame, [x1, y1, x2, y2])  # pyright: ignore
                        face_dict["spoof_real"] = sr.is_real
                        face_dict["spoof_confidence"] = float(sr.confidence)
                    except Exception:
                        pass
                else:
                    cached = self._match_label_cache(face_dict["bbox"])
                    if cached:
                        for key in (
                            "age",
                            "race",
                            "gender",
                            "emotion",
                            "spoof_real",
                            "spoof_confidence",
                        ):
                            if key in cached:
                                face_dict[key] = cached[key]

            faces.append(face_dict)

        # Consume background race results into persistent buffer
        if self._race_future is not None and self._race_future.done():
            try:
                results = self._race_future.result()
                if results:
                    self._race_results.extend(results)
                    self._race_results = self._race_results[-3:]  # keep freshest 3
            except Exception:
                pass
            self._race_future = None

        # Kick off background race inference (3x slower than main labels)
        if (
            show_labels
            and self._race is not None
            and self._race_future is None
            and self._frame_count % (config.INFERENCE_UPDATE_INTERVAL * 3) == 0
        ):
            face_data = [
                (f["bbox"], self._get_five_point_landmarks(f["landmarks"]))
                for f in faces
                if f.get("bbox") and f.get("landmarks")
            ]
            if face_data:
                self._race_future = self._race_executor.submit(
                    self._background_race_inference, frame.copy(), face_data
                )

        # Update cache on label inference frames
        if (
            show_labels
            and self._age_gender is not None
            and self._emotion is not None
            and self._spoofing is not None
            and self._frame_count % config.INFERENCE_UPDATE_INTERVAL == 0
        ):
            self._label_cache = []
            for face in faces:
                entry = {"bbox": face["bbox"]}
                for key in (
                    "age",
                    "race",
                    "gender",
                    "emotion",
                    "spoof_real",
                    "spoof_confidence",
                ):
                    if key in face:
                        entry[key] = face[key]
                self._label_cache.append(entry)
            self._label_cache = self._label_cache[-4:]

        # Apply persistent race results to every frame — matches by best IoU
        # so each person gets their own race label consistently.
        if self._race_results:
            for face in faces:
                best_iou = 0.0
                best_race = None
                for r_bbox, r_race in self._race_results:
                    iou = self._bbox_iou(face["bbox"], r_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_race = r_race
                if best_race is not None:
                    face["race"] = best_race

        return faces
