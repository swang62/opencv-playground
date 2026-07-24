"""Per-frame visual filters — OpenCV-only, no model needed."""

import cv2
import numpy as np


def apply_visual_filter(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "None":
        return frame

    if mode == "Sketch":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    if mode == "Thermal":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    if mode == "Invert":
        return 255 - frame

    if mode == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(frame, kernel).astype(np.uint8)

    if mode == "Emboss":
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        emboss = cv2.filter2D(gray, -1, kernel) + 128
        return cv2.cvtColor(emboss, cv2.COLOR_GRAY2BGR)

    if mode == "VHS Glitch":
        h, w = frame.shape[:2]
        result = frame.copy()
        shift = np.random.randint(-8, 9, 2).tolist()
        if shift[0] != 0:
            result[:, :, 2] = np.roll(result[:, :, 2], shift[0], axis=1)
        if shift[1] != 0:
            result[:, :, 0] = np.roll(result[:, :, 0], shift[1], axis=1)
        result[::3, :] = (result[::3, :] * 0.5).astype(np.uint8)
        return result

    if mode == "Comic":
        smooth = cv2.bilateralFilter(frame, 9, 75, 75)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 2)
        smooth[edges == 0] = (0, 0, 0)
        return smooth

    if mode == "Pixelate":
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 16, h // 16), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    return frame


FILTER_NAMES = [
    "None", "Sketch", "Thermal", "VHS Glitch", "Comic",
    "Emboss", "Invert", "Pixelate", "Sepia",
]
