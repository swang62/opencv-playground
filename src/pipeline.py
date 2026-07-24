"""Non-blocking webcam capture and inference pipeline."""

from __future__ import annotations

import logging
import threading
import time

import cv2

from src import config
from src.camera import create_camera
from src.detection import annotate_frame, extract_detections
from src.face import draw_face_mesh
from src.state import get_predict_kwargs

logger = logging.getLogger(__name__)


class CapturePipeline:
    """Non-blocking webcam capture and inference pipeline.

    Uses one thread to capture frames as fast as possible from a camera
    and a second thread to run inference on the *latest* frame, keeping
    at most one frame and one annotated JPEG in memory at any time.
    The pipeline never queues stale frames.

    Parameters
    ----------
    model : ModelBundle
        The model bundle used for inference.
    state : AppState
        Shared application state.
    camera : int or camera-like
        Camera index (default ``config.CAMERA_INDEX``) or any object
        with ``.read()`` returning ``(bool, frame)`` and ``.release()``.
    """

    def __init__(self, model, state, camera=config.CAMERA_INDEX):
        self.model = model
        self.state = state
        self._camera_input = camera
        self._camera = None
        self._latest_frame = None
        self._latest_encoded_frame = None
        self._frame_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._capture_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self):
        """Open the camera and start capture / inference threads."""
        logger.info("Starting capture pipeline (camera=%s)", self._camera_input)
        if isinstance(self._camera_input, int):
            try:
                cap = create_camera(self._camera_input)
            except RuntimeError as exc:
                msg = str(exc)
                logger.error("Camera init failed: %s", msg)
                self.error = msg
                self.state.set_camera_error(msg)
                self.state.set_camera_ready(False)
                return
        else:
            cap = self._camera_input
            if not cap.isOpened():
                msg = "Camera failed to open"
                logger.error(msg)
                self.error = msg
                self.state.set_camera_error(msg)
                self.state.set_camera_ready(False)
                return

        self._camera = cap
        self.state.set_camera_ready(True)
        self.state.set_camera_error(None)
        logger.info("Camera opened successfully")

        self._capture_thread = threading.Thread(
            target=self.capture_loop, daemon=True
        )
        self._inference_thread = threading.Thread(
            target=self.inference_loop, daemon=True
        )
        self._capture_thread.start()
        self._inference_thread.start()
        logger.info("Capture and inference threads started")

    def stop(self):
        """Signal shutdown, join threads, and release the camera."""
        logger.info("Stopping capture pipeline")
        self.state.signal_shutdown()

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5)
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=5)

        if self._camera is not None:
            self._camera.release()
            logger.info("Camera released")

    def get_latest_encoded_frame(self) -> bytes | None:
        """Return the most recent annotated image bytes, or None."""
        with self._result_lock:
            return self._latest_encoded_frame

    def capture_loop(self):
        cap = self._camera
        if cap is None:
            return
        logger.info("Capture loop started")
        while not self.state.shutdown:
            ret, frame = cap.read()
            if not ret:
                msg = "Camera read failed"
                logger.error(msg)
                self.error = msg
                self.state.set_camera_error(msg)
                self.state.set_camera_ready(False)
                break
            with self._frame_lock:
                self._latest_frame = cv2.flip(frame, 1)  # pyright: ignore[reportArgumentType, reportCallIssue]
        logger.info("Capture loop ended")

    def inference_loop(self):
        logger.info("Inference loop started")
        consecutive_errors = 0
        while not self.state.shutdown:
            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                time.sleep(0.005)
                continue

            t0 = time.perf_counter()

            # Capture mode once to avoid TOCTOU race: the UI thread can
            # switch modes between the predict call and the branch below.
            mode = self.state.mode

            try:
                kwargs = get_predict_kwargs(self.state)
                results = self.model.predict(frame, mode, **kwargs)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                logger.warning("Inference error (%d/5): %s", consecutive_errors, exc)
                if consecutive_errors >= 5:
                    msg = f"Model inference failed after {consecutive_errors} attempts: {exc}"
                    logger.error(msg)
                    self.error = msg
                    self.state.set_camera_error(msg)
                    self.state.set_camera_ready(False)
                    break
                time.sleep(0.1)
                continue

            if mode == "face":
                detections = results  # already extracted dicts from FaceEngine
                annotated = draw_face_mesh(
                    frame.copy(), detections, self.state.frames_per_second
                )
            else:
                detections = extract_detections(
                    results, self.state.confidence, need_masks=(mode == "find")
                )

                if mode == "everything":
                    label_best: dict[str, float] = {}
                    for d in detections:
                        lb = d["label"]
                        label_best[lb] = max(label_best.get(lb, 0.0), d["confidence"])
                    top_n = set(
                        sorted(label_best, key=lambda k: label_best[k], reverse=True)[
                            : self.state.top_labels
                        ]
                    )
                    detections = [d for d in detections if d["label"] in top_n]

                annotated = annotate_frame(
                    frame, detections, self.state.frames_per_second, mode
                )

            elapsed = time.perf_counter() - t0
            new_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            self.state.update_frames_per_second(new_fps)

            success, jpeg = cv2.imencode(".jpg", annotated)
            if success:
                with self._result_lock:
                    self._latest_encoded_frame = jpeg.tobytes()
        logger.info("Inference loop ended")
