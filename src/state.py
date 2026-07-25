"""Application state and query normalization."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from src import config


@dataclass
class AppState:
    """Thread-safe holder for mutable application state."""

    mode: str = "find"
    submitted_target: str = ""
    confidence: float = config.DEFAULT_THRESHOLD
    top_labels: int = 5
    camera_ready: bool = False
    models_ready: bool = False
    models_error: str | None = None
    camera_error: str | None = None
    shutdown: bool = False
    face_show_wireframe: bool = True
    face_show_labels: bool = False
    face_show_skeleton: bool = False
    face_show_ids: bool = False
    face_id_names: dict[int, str] = field(default_factory=dict)
    active_face_ids: set[int] = field(default_factory=set)
    privacy_mode: str = "None"
    visual_filter: str = "None"
    mask_opacity: float = config.DEFAULT_OPACITY
    overlay_color_name: str = "Green"
    font_scale: float = config.FONT_SCALE
    line_thickness: int = config.OVERLAY_THICKNESS
    roi_active: bool = False
    roi_x1: float = 0.0
    roi_y1: float = 0.0
    roi_x2: float = 0.0
    roi_y2: float = 0.0

    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def submit_target(self, target: str):
        with self._lock:
            self.submitted_target = target

    def set_mode(self, mode: str):
        with self._lock:
            self.mode = mode

    def set_confidence(self, value: float | None):
        if value is not None:
            with self._lock:
                self.confidence = value

    def set_camera_ready(self, value: bool):
        with self._lock:
            self.camera_ready = value

    def set_models_ready(self, value: bool):
        with self._lock:
            self.models_ready = value

    def set_camera_error(self, value: str | None):
        with self._lock:
            self.camera_error = value

    def set_models_error(self, value: str | None):
        with self._lock:
            self.models_error = value

    def signal_shutdown(self):
        with self._lock:
            self.shutdown = True

    def reset_shutdown(self):
        with self._lock:
            self.shutdown = False

    def set_active_face_ids(self, ids: set[int]):
        with self._lock:
            self.active_face_ids = ids

    def set_face_id_name(self, track_id: int, name: str):
        with self._lock:
            self.face_id_names[track_id] = name

    def clear_face_id_name(self, track_id: int):
        with self._lock:
            self.face_id_names.pop(track_id, None)

    def load_face_id_names(self):
        # Names are persisted in identities.json and reattached after
        # recognition; do not wipe the current session map on page refresh.
        return None

    def set_roi(self, x1: float, y1: float, x2: float, y2: float):
        with self._lock:
            self.roi_active = True
            self.roi_x1 = x1
            self.roi_y1 = y1
            self.roi_x2 = x2
            self.roi_y2 = y2

    def clear_roi(self):
        with self._lock:
            self.roi_active = False


def get_predict_kwargs(state: AppState):
    """Build keyword arguments for model.predict() based on current state."""
    is_find = state.mode == "find"
    conf = config.FIND_CONFIDENCE if is_find else state.confidence
    return {
        "conf": conf,
        "verbose": False,
        "imgsz": config.INFERENCE_SIZE,
        "retina_masks": is_find,
        "max_det": 10,
    }


COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "White": (255, 255, 255),
    "Black": (0, 0, 0),
    "Red": (0, 0, 255),
    "Yellow": (0, 255, 255),
    "Green": (0, 255, 0),
    "Cyan": (255, 255, 0),
    "Magenta": (255, 0, 255),
}


def color_name_to_bgr(name: str) -> tuple[int, int, int]:
    return COLOR_MAP.get(name, (0, 255, 0))  # default green


def color_name_to_hex(name: str) -> str:
    bgr = color_name_to_bgr(name)
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"
