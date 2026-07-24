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
    confidence: float = config.DEFAULT_CONFIDENCE
    top_labels: int = 5
    camera_ready: bool = False
    models_ready: bool = False
    models_error: str | None = None
    camera_error: str | None = None
    frames_per_second: float = 0.0
    shutdown: bool = False
    face_show_wireframe: bool = True
    face_show_headpose: bool = True
    face_show_labels: bool = True
    face_show_skeleton: bool = False
    privacy_mode: str = "None"
    visual_filter: str = "None"
    mask_opacity: float = config.MASK_OPACITY
    overlay_color_name: str = "Green"
    font_scale: float = config.FONT_SCALE
    line_thickness: int = config.OVERLAY_THICKNESS

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

    def update_frames_per_second(self, value: float):
        with self._lock:
            self.frames_per_second = value

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


def get_predict_kwargs(state: AppState):
    """Build keyword arguments for model.predict() based on current state."""
    is_find = state.mode == "find"
    conf = 0.1 if is_find else state.confidence
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
