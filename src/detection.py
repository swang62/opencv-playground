"""Detection extraction and frame annotation."""

from __future__ import annotations

import cv2
import numpy as np

from src import config

# Pre-allocated green overlay reused across annotate_frame calls.
_green_buf: np.ndarray | None = None


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
            bbox = tuple(float(v) for v in boxes.xyxy[i])
            det["bbox"] = bbox

            mask_tensor = masks_attr.data[i]
            if hasattr(mask_tensor, "cpu"):
                det["mask"] = mask_tensor.cpu().numpy()
            else:
                det["mask"] = np.asarray(mask_tensor)
        else:
            det["bbox"] = tuple(float(v) for v in boxes.xyxy[i])

        detections.append(det)

    return detections


def annotate_frame(frame, detections, fps: float, hidden_labels, mode: str = "find"):
    """Draw boxes, masks, and labels on a copy of *frame*.

    In ``"everything"`` mode: bounding boxes with labels.
    In ``"find"`` mode: green mask overlay with label at bbox center.
    Detections whose label appears in hidden_labels are skipped.
    """
    global _green_buf

    annotated = frame.copy()
    h, w = frame.shape[:2]

    if mode == "everything":
        for det in detections:
            if det["label"] in hidden_labels:
                continue
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, det["label"], (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.LABEL_FONT_SCALE, (0, 255, 0), config.LABEL_FONT_THICKNESS,
            )
    else:
        if _green_buf is None or _green_buf.shape[:2] != (h, w):
            _green_buf = np.zeros((h, w, 3), dtype=np.uint8)

        for det in detections:
            if det["label"] in hidden_labels:
                continue
            mask = det.get("mask")
            x1, y1, x2, y2 = map(int, det["bbox"])

            if mask is not None:
                mask_rz = cv2.resize(
                    mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
                )
                _green_buf.fill(0)
                _green_buf[mask_rz > 0.25] = (0, 200, 0)
                annotated = cv2.addWeighted(annotated, 0.9, _green_buf, 0.2, 0)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(
                annotated, det["label"], (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.LABEL_FONT_SCALE, (0, 255, 0), config.LABEL_FONT_THICKNESS,
            )

    cv2.putText(
        annotated, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FPS_FONT_SCALE, (0, 255, 0), config.FPS_FONT_THICKNESS,
    )
    return annotated
