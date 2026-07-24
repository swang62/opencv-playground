"""Detection extraction and frame annotation."""

from __future__ import annotations

import cv2
import numpy as np

from src import config

# Pre-allocated green overlay reused across annotate_frame calls.
green_buf: np.ndarray | None = None


def extract_detections(results, confidence: float, need_masks: bool = True):
    """Convert ultralytics Results objects to simple dicts for annotation.

    Each returned dict has keys: label, confidence, bbox (xyxy tuple),
    and mask (optional 2-D numpy array, only populated when *need_masks*).
    Detections below the confidence threshold are omitted.
    """
    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = getattr(result, "names", {})
    detections = []
    masks_attr = getattr(result, "masks", None) if need_masks else None

    for i in range(len(boxes.conf)):
        conf = float(boxes.conf[i])
        if conf < confidence:
            continue

        cls_id = int(boxes.cls[i])
        label = names.get(cls_id, str(cls_id))

        det = {"label": label, "confidence": conf, "mask": None}

        if masks_attr is not None:
            det["bbox"] = tuple(float(v) for v in boxes.xyxy[i])
            mask_tensor = masks_attr.data[i]
            if hasattr(mask_tensor, "cpu"):
                det["mask"] = mask_tensor.cpu().numpy()
            else:
                det["mask"] = np.asarray(mask_tensor)
        else:
            det["bbox"] = tuple(float(v) for v in boxes.xyxy[i])

        detections.append(det)

    return detections


def annotate_frame(frame, detections, fps: float, mode: str = "find", mask_opacity: float = config.MASK_OPACITY):
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
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2),
                config.OVERLAY_COLOR, config.OVERLAY_THICKNESS,
            )
            conf_pct = int(det["confidence"] * 100)
            cv2.putText(
                annotated, f"{det['label']} ({conf_pct}%)", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, config.OVERLAY_COLOR,
                config.FONT_THICKNESS,
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
                green_buf[mask_rz > 0.25] = config.OVERLAY_COLOR

        if green_buf.any():
            mask_region = green_buf.any(axis=2)
            annotated[mask_region] = cv2.addWeighted(
                annotated[mask_region], 1 - mask_opacity,
                green_buf[mask_region], mask_opacity, 0,
            )

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(
                annotated, det["label"], (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, config.OVERLAY_COLOR,
                config.FONT_THICKNESS,
            )

    return annotated
