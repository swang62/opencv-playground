"""Non-blocking webcam capture and inference pipeline."""

from __future__ import annotations

import logging
import threading
import time

import cv2

from src import config
from src.body import draw_hand_skeleton, draw_pose_skeleton
from src.camera import create_camera
from src.detection import annotate_frame, extract_detections
from src.face import apply_privacy, draw_face_mesh
from src.filters import apply_visual_filter
from src.state import color_name_to_bgr, get_predict_kwargs

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
        self._latest_frame_time = 0.0
        self._latest_encoded_frame = None
        self._frame_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._capture_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self.error: str | None = None

    def _get_first_frame(self, timeout_seconds: float = 5.0):
        deadline = time.time() + timeout_seconds
        while not self.state.shutdown and time.time() < deadline:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is not None:
                return frame.copy()
            time.sleep(0.01)
        return None

    def _prewarm_real_frame(self, frame):
        logger.info("Pre-warming real webcam frame paths...")

        try:
            for _ in range(3):
                self.model.face_engine.process(
                    frame,
                    show_headpose=False,
                    show_labels=False,
                )
        except Exception as exc:
            logger.warning("Real-frame face warmup failed: %s", exc)

        try:
            self.model.body_engine.process_pose(frame)
        except Exception as exc:
            logger.warning("Real-frame pose warmup failed: %s", exc)

        try:
            self.model.body_engine.process_hands(frame)
        except Exception as exc:
            logger.warning("Real-frame hand warmup failed: %s", exc)

        logger.info("Real webcam frame warmup complete")

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

        self._capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self._capture_thread.start()

        first_frame = self._get_first_frame()
        if first_frame is not None:
            self._prewarm_real_frame(first_frame)
        else:
            logger.warning(
                "Timed out waiting for first camera frame; skipping real-frame warmup"
            )

        self._inference_thread = threading.Thread(
            target=self.inference_loop, daemon=True
        )
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
            now = time.time()
            with self._frame_lock:
                assert frame is not None
                self._latest_frame = cv2.flip(frame, 1)
                self._latest_frame_time = now
        logger.info("Capture loop ended")

    def inference_loop(self):
        logger.info("Inference loop started")
        consecutive_errors = 0
        stall_warned = False
        while not self.state.shutdown:
            with self._frame_lock:
                frame = self._latest_frame
                frame_time = self._latest_frame_time

            if frame is None:
                time.sleep(0.005)
                continue

            # If display sleep has frozen the camera stream, skip inference
            # so we don't busy-loop on a stale frame.
            if time.time() - frame_time > 1.5:
                if not stall_warned:
                    logger.info("Camera stream stalled (display sleep?) — waiting")
                    stall_warned = True
                time.sleep(0.05)
                continue
            stall_warned = False

            # Capture mode once to avoid TOCTOU race: the UI thread can
            # switch modes between the predict call and the branch below.
            mode = self.state.mode

            try:
                kwargs = get_predict_kwargs(self.state)
                if mode == "face":
                    kwargs["show_headpose"] = self.state.face_show_headpose
                    kwargs["show_labels"] = self.state.face_show_labels
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

            oc = color_name_to_bgr(self.state.overlay_color_name)
            ft = max(2, int(self.state.font_scale * 1.5))

            if mode == "face":
                detections = results  # already extracted dicts from FaceEngine
                show_headpose = self.state.face_show_headpose
                show_labels = self.state.face_show_labels
                draw_frame = frame.copy()
                draw_frame = apply_visual_filter(
                    draw_frame,
                    self.state.visual_filter,
                )
                annotated = draw_face_mesh(
                    draw_frame,
                    detections,
                    show_wireframe=self.state.face_show_wireframe,
                    show_headpose=show_headpose,
                    show_labels=show_labels,
                    overlay_color=oc,
                    font_scale=self.state.font_scale,
                    font_thickness=ft,
                    line_thickness=self.state.line_thickness,
                )
                privacy_mode = self.state.privacy_mode
                if privacy_mode != "None":
                    annotated = apply_privacy(
                        annotated,
                        detections,
                        privacy_mode,
                        inplace=True,
                    )

                if self.state.face_show_skeleton:
                    try:
                        poses = self.model.body_engine.process_pose(frame)
                        for pts in poses:
                            draw_pose_skeleton(
                                annotated,
                                pts,
                                color=oc,
                                thickness=self.state.line_thickness,
                                joint_radius=max(3, self.state.line_thickness + 2),
                            )
                    except Exception:
                        pass

                    try:
                        hands = self.model.body_engine.process_hands(frame)
                        for pts in hands:
                            draw_hand_skeleton(
                                annotated,
                                pts,
                                color=oc,
                                thickness=self.state.line_thickness,
                                joint_radius=max(3, self.state.line_thickness + 2),
                            )
                    except Exception:
                        pass

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

                draw_frame = apply_visual_filter(
                    frame.copy(),
                    self.state.visual_filter,
                )
                annotated = annotate_frame(
                    draw_frame,
                    detections,
                    mode,
                    mask_opacity=self.state.mask_opacity,
                    overlay_color=oc,
                    font_scale=self.state.font_scale,
                    font_thickness=ft,
                    line_thickness=self.state.line_thickness,
                )

            success, jpeg = cv2.imencode(".jpg", annotated)
            if success:
                with self._result_lock:
                    self._latest_encoded_frame = jpeg.tobytes()
        logger.info("Inference loop ended")
