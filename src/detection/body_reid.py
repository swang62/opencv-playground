"""Body Identity Engine — YOLO11n person tracking + OSNet Re-ID."""

from __future__ import annotations

import logging
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from uniface.stores import FAISS

from src import config

logger = logging.getLogger(__name__)

# OSNet x1.0 input dimensions (from model graph: [1, 3, 256, 128])
_OSNET_HEIGHT = 256
_OSNET_WIDTH = 128

# ImageNet normalization stats used by osnet_x1_0
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BodyIdEngine:
    """Lazy-loaded YOLO11 person tracker + OSNet body re-identification.

    Maintains a separate FAISS store for body embeddings, a per-track
    identity cache, and snapshots of the latest frame+detection for each
    active track (for enrollment and thumbnail extraction).

    Thread-safe via internal lock on model loading; intended to be called
    from a single worker thread at runtime.
    """

    def __init__(self):
        self._detector = None  # ultralytics.YOLO
        self._session = None  # onnxruntime.InferenceSession
        self._lock = threading.Lock()
        self._store = None  # FAISS gallery
        self._track_identity_cache: dict[int, dict[str, str]] = {}
        self._track_snapshots: dict[int, dict] = {}
        self._track_embedding_buffers: dict[int, deque] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_loaded(self):
        """Lazy-load YOLO detector and OSNet ONNX session (idempotent)."""
        if self._detector is not None and self._session is not None:
            return
        with self._lock:
            if self._detector is not None and self._session is not None:
                return
            self._load_detector()
            self._load_onnx_session()

    def warmup(self):
        """Run a dummy inference to trigger model loading + one-time JIT."""
        self.ensure_loaded()
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            self.track_and_recognize(dummy)
        except Exception as exc:
            logger.warning("BodyIdEngine warmup failed: %s", exc)

    def track_and_recognize(self, frame: np.ndarray) -> list[dict]:
        """Run person tracking, recognize unseen tracks, return result dicts.

        Each result dict::

            {
                "label": "Person",
                "confidence": 0.92,
                "bbox": (x1, y1, x2, y2),  # int pixel coords
                "track_id": 1,
                "identity_name": "Alice",  # only present if recognized
            }

        Returns empty list on no persons or error.
        """
        self.ensure_loaded()
        assert self._detector is not None

        try:
            results = self._detector.track(
                frame,
                classes=[0],  # COCO class 0 = person
                conf=config.DETECT_CONFIDENCE,
                imgsz=config.INFERENCE_SIZE,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
        except Exception as exc:
            logger.warning("YOLO tracking failed: %s", exc)
            return []

        detections = _parse_tracking_results(results)

        # Filter out detections below minimum area ratio (hand/limb-only rejects)
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * config.IDENTITY_MIN_AREA_RATIO
        detections = [
            det
            for det in detections
            if det.get("bbox")
            and len(det["bbox"]) >= 4
            and (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])
            >= min_area
        ]

        # Snapshot every active detection for later enrollment/thumbnail use
        active_track_ids = set()
        for det in detections:
            tid = det["track_id"]
            active_track_ids.add(tid)
            self._track_snapshots[tid] = {
                "frame": frame.copy(),
                "detection": det,
            }

        # Purge entries for departed tracks
        stale_ids = set(self._track_identity_cache) - active_track_ids
        for tid in stale_ids:
            self._track_identity_cache.pop(tid, None)
        stale_snapshots = set(self._track_snapshots) - active_track_ids
        for tid in stale_snapshots:
            self._track_snapshots.pop(tid, None)
        stale_buffers = set(self._track_embedding_buffers) - active_track_ids
        for tid in stale_buffers:
            self._track_embedding_buffers.pop(tid, None)

        for det in detections:
            tid = det["track_id"]

            cached = self._track_identity_cache.get(tid)
            if cached is not None:
                det["identity_name"] = cached["identity_name"]
                continue

            crop = _crop_person(frame, det["bbox"])
            if crop is None:
                continue
            embedding = self._embed(crop)
            if embedding is None:
                continue

            buf = self._track_embedding_buffers.setdefault(
                tid, deque(maxlen=config.REID_EMBEDDING_BUFFER_SIZE)
            )
            buf.append(embedding)
            if len(buf) < config.REID_EMBEDDING_BUFFER_SIZE:
                continue

            mean_emb = np.mean(list(buf), axis=0)
            recognized = self._search_store(mean_emb)
            if recognized is not None:
                self._track_identity_cache[tid] = recognized
                if recognized.get("identity_name"):
                    det["identity_name"] = recognized["identity_name"]

        return detections

    def get_snapshot(self, track_id: int) -> dict | None:
        """Return the latest (frame, detection) snapshot for *track_id*, or None."""
        return self._track_snapshots.get(track_id)

    def enroll_identity(
        self, name: str, frame: np.ndarray, detection: dict, track_id: int
    ):
        """Enroll a body identity: normalize, store embedding, cache name.

        The embedding is computed from the *detection* bbox crop of *frame*,
        added to the FAISS body gallery (replacing any prior entry with the
        same *name*), and immediately associated with *track_id*.
        """
        self.ensure_loaded()

        crop = _crop_person(frame, detection["bbox"])
        if crop is None:
            logger.warning(
                "Cannot enroll body identity: invalid crop for track=%s", track_id
            )
            return

        embedding = self._embed(crop)
        if embedding is None:
            logger.warning(
                "Cannot enroll body identity: embedding failed for track=%s", track_id
            )
            return

        buf = self._track_embedding_buffers.get(track_id)
        if buf:
            all_embs = list(buf) + [embedding]
            embedding = np.mean(all_embs, axis=0)

        store = self._ensure_store()
        store.remove("name", name)
        store.add(embedding, {"name": name})
        store.save()

        self._track_identity_cache[track_id] = {"identity_name": name}
        logger.info(
            "Saved body identity: name=%s track=%s total=%d",
            name,
            track_id,
            len(store),
        )

    def remove_identity(self, name: str, track_id: int | None = None):
        """Remove a body identity from the gallery and track cache.

        When *track_id* is given only that track's cache entry is removed;
        otherwise every cached entry with the matching *name* is cleared.
        """
        store = self._ensure_store()
        removed = store.remove("name", name)
        if removed:
            store.save()
        if track_id is not None:
            self._track_identity_cache.pop(track_id, None)
            self._track_embedding_buffers.pop(track_id, None)
        else:
            self._track_identity_cache = {
                tid: ident
                for tid, ident in self._track_identity_cache.items()
                if ident.get("identity_name") != name
            }
            self._track_embedding_buffers.clear()

    # ------------------------------------------------------------------
    # Internal — model loading
    # ------------------------------------------------------------------

    def _load_detector(self):
        from ultralytics import YOLO  # lazy import

        self._detector = YOLO(config.BODY_DETECTION_MODEL)
        logger.info("YOLO11 person detector loaded")

    def _load_onnx_session(self):
        import onnxruntime as ort  # lazy import

        model_path = Path(config.BODY_REID_MODEL)
        if not model_path.exists():
            msg = f"Body Re-ID model not found at {model_path}. Run 'uv run python scripts/prepare_body_reid_model.py' first."
            raise FileNotFoundError(msg)
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        logger.info("OSNet ONNX session created (CoreML)")

    def _ensure_store(self):
        if self._store is not None:
            return self._store
        with self._lock:
            if self._store is not None:
                return self._store
            body_dir = Path(config.BODY_IDENTITIES_DIR)
            body_dir.mkdir(parents=True, exist_ok=True)
            store = FAISS(db_path=str(body_dir))
            loaded = store.load()
            logger.info(
                "Body identity store ready: path=%s loaded=%s size=%d",
                body_dir,
                loaded,
                len(store),
            )
            if not loaded:
                logger.info(
                    "Body identity store is empty; enroll a body to create entries"
                )
            self._store = store
            return self._store

    # ------------------------------------------------------------------
    # Internal — embedding
    # ------------------------------------------------------------------

    def _embed(self, crop: np.ndarray) -> np.ndarray | None:
        """Compute L2-normalised 512-D OSNet embedding for a person crop.

        Preprocessing: BGR→RGB → resize to 256×128 → float32 [0,1] →
        ImageNet normalise → NCHW. The raw ONNX output is L2-normalised
        before returning.
        """
        assert self._session is not None

        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (_OSNET_WIDTH, _OSNET_HEIGHT))

            # Normalise to [0,1] then apply ImageNet stats
            blob = resized.astype(np.float32) / 255.0
            blob = (blob - _MEAN) / _STD
            blob = blob.transpose(2, 0, 1)  # HWC → CHW
            blob = np.expand_dims(blob, axis=0)  # → NCHW

            input_name = self._session.get_inputs()[0].name
            output_name = self._session.get_outputs()[0].name
            raw = self._session.run([output_name], {input_name: blob})[0]
            assert isinstance(raw, np.ndarray)
            embedding = raw.ravel().astype(np.float32)

            # L2 normalise (FAISS IndexFlatIP uses cosine sim = inner product)
            norm = np.linalg.norm(embedding)
            if norm > 1e-6:
                embedding = embedding / norm
            return embedding
        except Exception as exc:
            logger.warning("OSNet embedding failed: %s", exc)
            return None

    def _search_store(self, embedding: np.ndarray) -> dict | None:
        """Search the body FAISS gallery for *embedding*.

        Returns ``{"identity_name": name}`` on match, or None.
        """
        store = self._ensure_store()
        if len(store) == 0:
            return None

        result, similarity = store.search(
            embedding,
            threshold=config.SIMILARITY_THRESHOLD,
        )
        if result is None:
            return None

        matched_name = str(result.get("name", ""))
        logger.debug(
            "Body recognition matched name=%s score=%.4f",
            matched_name,
            similarity,
        )
        return {"identity_name": matched_name}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_tracking_results(results) -> list[dict]:
    """Parse Ultralytics tracking results into plain dicts.

    Each dict::

        {
            "label": "Person",
            "confidence": 0.92,
            "bbox": (x1, y1, x2, y2),
            "track_id": 1,
        }

    Returns empty list when no persons detected or tracking IDs are absent.
    """
    detections = []
    for result in results:
        if result.boxes is None or result.boxes.id is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)
        confs = result.boxes.conf.cpu().numpy()  # (N,)
        ids = result.boxes.id.cpu().numpy().astype(int)  # (N,)

        for i in range(len(boxes)):
            detections.append(
                {
                    "label": "Person",
                    "confidence": float(confs[i]),
                    "bbox": (
                        int(boxes[i][0]),
                        int(boxes[i][1]),
                        int(boxes[i][2]),
                        int(boxes[i][3]),
                    ),
                    "track_id": int(ids[i]),
                }
            )
    return detections


def _crop_person(frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
    """Extract and validate a person crop from the frame.

    Clamps coordinates to frame boundaries and rejects crops smaller than
    16×16 pixels.  Returns None for invalid or empty crops.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 - x1 < 16 or y2 - y1 < 16:
        return None

    crop = frame[y1:y2, x1:x2]
    return None if crop.size == 0 else crop
