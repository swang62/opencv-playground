"""Per-frame visual filters — OpenCV-only, no model needed."""

import cv2
import numpy as np


def apply_visual_filter(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "None":
        return frame

    if mode == "Sketch":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(255 - gray, (15, 15), 0)
        sketch = cv2.divide(gray, 255 - blur + 1, scale=200)
        return cv2.cvtColor(sketch.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if mode == "Thermal":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    if mode == "Invert":
        return 255 - frame

    if mode == "CRT":
        h, w = frame.shape[:2]
        result = frame.copy()
        result[:, :, 2] = np.roll(result[:, :, 2], 4, axis=1)
        result[:, :, 0] = np.roll(result[:, :, 0], -3, axis=1)
        scanlines = np.zeros((h, w, 3), dtype=np.uint8)
        scanlines[1::2, :] = 50
        result = (
            cv2.subtract(result.astype(np.int16), scanlines.astype(np.int16))
            .clip(0, 255)
            .astype(np.uint8)
        )
        return result

    if mode == "Comic":
        smooth = cv2.bilateralFilter(frame, 9, 75, 75)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
        )
        smooth[edges == 0] = (0, 0, 0)
        return smooth

    return frame


FILTER_NAMES = [
    "None",
    "Sketch",
    "Thermal",
    "CRT",
    "Comic",
    "Invert",
]
