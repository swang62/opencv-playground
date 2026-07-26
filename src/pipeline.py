"""Non-blocking webcam capture and inference pipeline."""

from __future__ import annotations

import logging
import threading
import time

import cv2

from src import config
from src.body import draw_hand_skeleton, draw_pose_skeleton
from src.camera import create_camera
from src.detection import annotate_frame, draw_body_boxes, extract_detections
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
        self._latest_display_frame = None
        self._latest_encoded_frame = None
        self._latest_face_detections: list[dict] = []
        self._latest_face_identity_results: list[dict] = []
        self._latest_body_identity_results: list[dict] = []
        self._face_id_input_frame = None
        self._face_id_input_detections: list[dict] = []
        self._face_id_input_time = 0.0
        self._body_id_input_frame = None
        self._body_id_input_time = 0.0
        self._frame_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._face_id_lock = threading.Lock()
        self._body_id_lock = threading.Lock()
        self._capture_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._face_id_thread: threading.Thread | None = None
        self._body_id_thread: threading.Thread | None = None
        self._detect_frame_num = 0
        self._cached_top_labels: set[str] = set()
        self._cached_label_detections: dict[str, list[dict]] = {}
        self._cached_label_last_seen: dict[str, int] = {}
        self._last_top_k: int = 5
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
        self.state.reset_shutdown()
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
        self._face_id_thread = threading.Thread(target=self.face_id_loop, daemon=True)
        self._body_id_thread = threading.Thread(target=self.body_id_loop, daemon=True)
        self._inference_thread.start()
        self._face_id_thread.start()
        self._body_id_thread.start()
        logger.info("Capture, inference, and identity threads started")

    def stop(self):
        """Signal shutdown, join threads, and release the camera."""
        logger.info("Stopping capture pipeline")
        self.state.signal_shutdown()

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5)
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=5)
        if self._face_id_thread is not None:
            self._face_id_thread.join(timeout=5)
        if self._body_id_thread is not None:
            self._body_id_thread.join(timeout=5)

        if self._camera is not None:
            self._camera.release()
            logger.info("Camera released")
        self.state.set_camera_ready(False)

    def get_latest_encoded_frame(self) -> bytes | None:
        """Return the most recent annotated image bytes, or None."""
        with self._result_lock:
            return self._latest_encoded_frame

    def get_latest_frame_copy(self):
        with self._result_lock:
            display_frame = self._latest_display_frame
        if display_frame is not None:
            return display_frame.copy()
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_latest_face_detection(self, track_id: int):
        with self._result_lock:
            for detection in self._latest_face_detections:
                if detection.get("track_id") == track_id:
                    return detection.copy()
            for detection in self._latest_face_identity_results:
                if detection.get("track_id") == track_id:
                    return detection.copy()
        return None

    def face_id_loop(self):
        logger.info("Face ID loop started")
        last_processed_time = 0.0
        while not self.state.shutdown:
            with self._face_id_lock:
                request_time = self._face_id_input_time
                frame = (
                    None
                    if self._face_id_input_frame is None
                    else self._face_id_input_frame.copy()
                )
                detections = [
                    detection.copy() for detection in self._face_id_input_detections
                ]

            if request_time <= last_processed_time or frame is None or not detections:
                time.sleep(0.01)
                continue

            try:
                identity_results = self.model.face_engine.apply_face_identities(
                    frame, detections
                )
                current_ids = {
                    detection["track_id"]
                    for detection in identity_results
                    if "track_id" in detection
                }
                for detection in identity_results:
                    track_id = detection.get("track_id")
                    identity_name = detection.get("identity_name")
                    if track_id is not None and isinstance(identity_name, str):
                        self.state.set_face_id_name(track_id, identity_name)
                self.state.set_active_face_ids(current_ids)
                with self._result_lock:
                    self._latest_face_identity_results = [
                        detection.copy() for detection in identity_results
                    ]
            except Exception as exc:
                logger.warning("Face ID worker error: %s", exc)

            last_processed_time = request_time
        logger.info("Face ID loop ended")

    def body_id_loop(self):
        logger.info("Body ID loop started")
        last_processed_time = 0.0
        while not self.state.shutdown:
            with self._body_id_lock:
                request_time = self._body_id_input_time
                frame = (
                    None
                    if self._body_id_input_frame is None
                    else self._body_id_input_frame.copy()
                )

            if request_time <= last_processed_time or frame is None:
                time.sleep(0.01)
                continue

            try:
                body_results = self.model.body_id_engine.track_and_recognize(frame)
                current_ids = {
                    det["track_id"] for det in body_results if "track_id" in det
                }
                for det in body_results:
                    track_id = det.get("track_id")
                    identity_name = det.get("identity_name")
                    if track_id is not None and isinstance(identity_name, str):
                        self.state.set_body_id_name(track_id, identity_name)
                self.state.set_active_body_ids(current_ids)
                with self._result_lock:
                    self._latest_body_identity_results = [
                        det.copy() for det in body_results
                    ]
            except Exception as exc:
                logger.warning("Body ID worker error: %s", exc)

            last_processed_time = request_time
        logger.info("Body ID loop ended")

    def get_latest_body_snapshot(self, track_id: int) -> dict | None:
        """Return the latest (frame, detection) snapshot for a body track, or None."""
        return self.model.body_id_engine.get_snapshot(track_id)

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

            # ROI crop: if active, run inference and display on the selected region.
            display_frame = frame
            if self.state.roi_active:
                h, w = frame.shape[:2]
                rx1 = max(0, min(int(self.state.roi_x1), int(self.state.roi_x2)))
                ry1 = max(0, min(int(self.state.roi_y1), int(self.state.roi_y2)))
                rx2 = min(w, max(int(self.state.roi_x1), int(self.state.roi_x2)))
                ry2 = min(h, max(int(self.state.roi_y1), int(self.state.roi_y2)))
                if (rx2 - rx1) >= 16 and (ry2 - ry1) >= 16:
                    display_frame = cv2.resize(
                        frame[ry1:ry2, rx1:rx2].copy(),
                        (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )

            try:
                kwargs = get_predict_kwargs(self.state)
                if mode == "face":
                    kwargs["show_labels"] = self.state.tracking_enabled
                results = self.model.predict(display_frame, mode, **kwargs)
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
            ft = max(2, int((self.state.font_scale or config.FONT_SCALE) * 1.5))

            if mode == "face":
                detections = results  # already extracted dicts from FaceEngine
                show_labels = self.state.tracking_enabled
                if self.state.tracking_enabled:
                    # Feed face identity worker
                    with self._face_id_lock:
                        self._face_id_input_frame = display_frame.copy()
                        self._face_id_input_detections = [
                            detection.copy() for detection in detections
                        ]
                        self._face_id_input_time = frame_time
                    # Feed body identity worker (frame only; BodyIdEngine does its own detection)
                    with self._body_id_lock:
                        self._body_id_input_frame = display_frame.copy()
                        self._body_id_input_time = frame_time
                    # Merge face identity results
                    if self.state.tracking_enabled:
                        with self._result_lock:
                            identity_results = [
                                detection.copy()
                                for detection in self._latest_face_identity_results
                            ]
                        self.model.face_engine.merge_face_identity_results(
                            detections,
                            identity_results,
                        )
                else:
                    # Clear face identity worker
                    with self._face_id_lock:
                        self._face_id_input_frame = None
                        self._face_id_input_detections = []
                        self._face_id_input_time = 0.0
                    with self._result_lock:
                        self._latest_face_identity_results = []
                    self.state.set_active_face_ids(set())
                    # Clear body identity worker
                    with self._body_id_lock:
                        self._body_id_input_frame = None
                        self._body_id_input_time = 0.0
                    with self._result_lock:
                        self._latest_body_identity_results = []
                    self.state.set_active_body_ids(set())
                with self._result_lock:
                    self._latest_face_detections = [
                        detection.copy() for detection in detections
                    ]
                draw_frame = display_frame.copy()
                draw_frame = apply_visual_filter(
                    draw_frame,
                    self.state.visual_filter,
                )
                # Snapshot the name dict for thread safety.
                name_snapshot = (
                    dict(self.state.face_id_names)
                    if self.state.tracking_enabled
                    else {}
                )
                annotated = draw_face_mesh(
                    draw_frame,
                    detections,
                    show_wireframe=self.state.face_mesh_enabled,
                    show_labels=show_labels,
                    overlay_color=oc,
                    font_scale=self.state.font_scale,
                    font_thickness=ft,
                    line_thickness=self.state.line_thickness,
                    face_id_names=name_snapshot,
                )
                privacy_mode = self.state.privacy_mode
                if privacy_mode != "None":
                    annotated = apply_privacy(
                        annotated,
                        detections,
                        privacy_mode,
                        inplace=True,
                    )

                if self.state.body_mesh_enabled:
                    try:
                        poses = self.model.body_engine.process_pose(display_frame)
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
                        hands = self.model.body_engine.process_hands(display_frame)
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

                # Draw faint body outlines when tracking is enabled
                if self.state.tracking_enabled:
                    with self._result_lock:
                        body_results = [
                            det.copy() for det in self._latest_body_identity_results
                        ]
                    if body_results:
                        try:
                            draw_body_boxes(
                                annotated,
                                body_results,
                                overlay_color=oc,
                                thickness=self.state.line_thickness,
                            )
                        except Exception:
                            pass

            else:
                # Non-face mode: ensure body identity worker is cleared
                with self._body_id_lock:
                    self._body_id_input_frame = None
                    self._body_id_input_time = 0.0
                with self._result_lock:
                    self._latest_body_identity_results = []
                self.state.set_active_body_ids(set())

                extraction_confidence = (
                    config.FIND_CONFIDENCE if mode == "find" else self.state.confidence
                )
                detections = extract_detections(
                    results,
                    extraction_confidence,
                    need_masks=(mode == "find"),
                )

                if mode == "everything":
                    frame_height, frame_width = display_frame.shape[:2]
                    max_box_area = (
                        frame_width * frame_height * config.MAX_DETECT_BOX_AREA_RATIO
                    )
                    detections = [
                        detection
                        for detection in detections
                        if (detection["bbox"][2] - detection["bbox"][0])
                        * (detection["bbox"][3] - detection["bbox"][1])
                        <= max_box_area
                    ]

                    self._detect_frame_num += 1

                    # Periodically recalculate which labels are top-N
                    top_k_changed = self.state.top_labels != self._last_top_k
                    if (
                        not self._cached_top_labels
                        or top_k_changed
                        or self._detect_frame_num % config.CAMERA_UPDATE_INTERVAL == 0
                    ):
                        label_best: dict[str, float] = {}
                        for d in detections:
                            lb = d["label"]
                            label_best[lb] = max(
                                label_best.get(lb, 0.0), d["confidence"]
                            )
                        self._cached_top_labels = set(
                            sorted(
                                label_best,
                                key=lambda k: label_best[k],
                                reverse=True,
                            )[: self.state.top_labels]
                        )
                        self._last_top_k = self.state.top_labels

                    # Keep cache limited to the current top-N labels so the
                    # slider directly controls what can be shown.
                    self._cached_label_detections = {
                        label: boxes
                        for label, boxes in self._cached_label_detections.items()
                        if label in self._cached_top_labels
                    }
                    self._cached_label_last_seen = {
                        label: frame_num
                        for label, frame_num in self._cached_label_last_seen.items()
                        if label in self._cached_top_labels
                    }

                    # Group current frame detections by label
                    current_by_label: dict[str, list[dict]] = {}
                    for d in detections:
                        if d["label"] in self._cached_top_labels:
                            current_by_label.setdefault(d["label"], []).append(d)

                    # Update cache with fresh detections for current top-N labels
                    for label, boxes in current_by_label.items():
                        self._cached_label_detections[label] = boxes
                        self._cached_label_last_seen[label] = self._detect_frame_num

                    # Temporal smoothing only within the current top-N set.
                    merged = []
                    for label in list(self._cached_top_labels):
                        if label in current_by_label:
                            merged.extend(current_by_label[label])
                        elif (
                            label in self._cached_label_detections
                            and self._detect_frame_num
                            - self._cached_label_last_seen.get(label, 0)
                            < 10
                        ):
                            merged.extend(self._cached_label_detections[label])
                        else:
                            self._cached_label_detections.pop(label, None)
                            self._cached_label_last_seen.pop(label, None)
                    detections = merged

                draw_frame = apply_visual_filter(
                    display_frame.copy(), self.state.visual_filter
                )
                annotated = annotate_frame(
                    draw_frame,
                    detections,
                    mode,
                    mask_opacity=0.3,
                    overlay_color=oc,
                    font_scale=self.state.font_scale,
                    font_thickness=ft,
                    line_thickness=self.state.line_thickness,
                )

            success, jpeg = cv2.imencode(".jpg", annotated)
            if success:
                with self._result_lock:
                    self._latest_display_frame = display_frame.copy()
                    self._latest_encoded_frame = jpeg.tobytes()
        logger.info("Inference loop ended")
