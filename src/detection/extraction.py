"""Detection result extraction — convert ultralytics Results to simple dicts."""

from __future__ import annotations

import numpy as np


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
