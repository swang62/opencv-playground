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
    prompt_busy: bool = False
    hidden_labels: set[str] = field(default_factory=set)
    latest_labels: dict[str, tuple[int, float]] = field(default_factory=dict)
    camera_ready: bool = False
    models_ready: bool = False
    models_error: str | None = None
    camera_error: str | None = None
    fps: float = 0.0
    shutdown: bool = False

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

    def toggle_hidden_label(self, label: str):
        with self._lock:
            if label in self.hidden_labels:
                self.hidden_labels = self.hidden_labels - {label}
            else:
                self.hidden_labels = self.hidden_labels | {label}

    def update_latest_labels(self, labels: dict[str, tuple[int, float]]):
        with self._lock:
            self.latest_labels = labels

    def update_fps(self, value: float):
        with self._lock:
            self.fps = value

    def set_camera_ready(self, value: bool):
        with self._lock:
            self.camera_ready = value

    def set_models_ready(self, value: bool):
        with self._lock:
            self.models_ready = value

    def set_models_error(self, value: str | None):
        with self._lock:
            self.models_error = value

    def set_camera_error(self, value: str | None):
        with self._lock:
            self.camera_error = value

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
