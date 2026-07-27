"""Frame annotation and body-box drawing."""

from __future__ import annotations

import cv2
import numpy as np

from src import config

# Pre-allocated green overlay reused across annotate_frame calls.
green_buf: np.ndarray | None = None


def draw_corner_bbox(img, x1, y1, x2, y2, color, thickness):
    """Corner brackets: thick L-shapes at corners, thin edge lines."""
    bw, bh = x2 - x1, y2 - y1
    cl = max(10, min(bw, bh) // 4)
    cl = min(cl, bw // 2, bh // 2, 40)
    et = max(1, thickness // 2)
    # Thin edges
    cv2.line(img, (x1 + cl, y1), (x2 - cl, y1), color, et)
    cv2.line(img, (x1, y1 + cl), (x1, y2 - cl), color, et)
    cv2.line(img, (x2, y1 + cl), (x2, y2 - cl), color, et)
    cv2.line(img, (x1 + cl, y2), (x2 - cl, y2), color, et)
    # Thick corner L-brackets
    for cx, cy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
        dx = cl if cx == x1 else -cl
        dy = cl if cy == y1 else -cl
        cv2.line(img, (cx, cy), (cx + dx, cy), color, thickness)
        cv2.line(img, (cx, cy), (cx, cy + dy), color, thickness)


def annotate_frame(
    frame,
    detections,
    mode: str = "find",
    mask_opacity: float = config.ALPHA,
    overlay_color=config.OVERLAY_COLOR,
    font_scale: float = config.FONT_SCALE,
    font_thickness: int = config.FONT_THICKNESS,
    line_thickness: int = config.OVERLAY_THICKNESS,
):
    """Draw boxes, masks, and labels on a copy of *frame*.

    In ``"everything"`` mode: bounding boxes with labels.
    In ``"find"`` mode: green mask overlay with label at bbox center.
    """
    global green_buf

    annotated = frame.copy()
    h, w = frame.shape[:2]

    if mode == "everything":
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            draw_corner_bbox(annotated, x1, y1, x2, y2, overlay_color, line_thickness)
            conf_pct = int(det["confidence"] * 100)
            cv2.putText(
                annotated,
                f"{det['label']} ({conf_pct}%)",
                (x1, y1 - 6),
                config.OVERLAY_FONT,
                font_scale,
                overlay_color,
                font_thickness,
            )
    else:
        if green_buf is None or green_buf.shape[:2] != (h, w):
            green_buf = np.zeros((h, w, 3), dtype=np.uint8)
        green_buf.fill(0)

        for det in detections:
            mask = det.get("mask")
            x1, y1, x2, y2 = map(int, det["bbox"])

            if mask is not None:
                mask_rz = cv2.resize(
                    mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
                )
                green_buf[mask_rz > 0.25] = overlay_color

        if green_buf.any():
            mask_region = green_buf.any(axis=2)
            annotated[mask_region] = cv2.addWeighted(
                annotated[mask_region],
                1 - mask_opacity,
                green_buf[mask_region],
                mask_opacity,
                0,
            )

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.putText(
                annotated,
                det["label"],
                (x1, y1 - 6),
                config.OVERLAY_FONT,
                font_scale,
                overlay_color,
                font_thickness,
            )

    return annotated


def draw_body_boxes(
    frame: np.ndarray,
    body_results: list[dict],
    overlay_color=config.OVERLAY_COLOR,
    thickness: int = config.OVERLAY_THICKNESS,
    font_scale: float = config.FONT_SCALE,
) -> np.ndarray:
    """Draw body bounding boxes with ID labels directly on the frame.

    Each detection gets a corner bracket box and an ``ID: N`` label.
    No alpha blending.
    """
    if not body_results:
        return frame

    for det in body_results:
        bbox = det.get("bbox")
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = map(int, bbox[:4])
        draw_corner_bbox(frame, x1, y1, x2, y2, overlay_color, thickness)
        track_id = det.get("track_id")
        identity_name = det.get("identity_name")
        if identity_name:
            label = identity_name
        elif track_id is not None:
            label = f"ID: {track_id}"
        else:
            label = ""
        if label:
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 4, 12)),
                config.OVERLAY_FONT,
                font_scale,
                (0, 255, 255),
                config.FONT_THICKNESS,
            )

    return frame
