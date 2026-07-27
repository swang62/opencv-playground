"""Face mesh drawing and privacy blur for face detections."""

from __future__ import annotations

import cv2
import numpy as np
from uniface.privacy import BlurFace

from src import config

# ---------------------------------------------------------------------------
# Face mesh connections — (start, end) index pairs per feature from the
# MediaPipe Face Mesh canonical 478-point topology (FaceLandmarksConnections).
# Using connection pairs (not ordered point lists) because several features
# have disconnected segments that polylines would bridge incorrectly.
# ---------------------------------------------------------------------------

FACE_OVAL = [
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10),
]

LEFT_EYE = [
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
]

RIGHT_EYE = [
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
]

LEFT_BROW = [
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
]

RIGHT_BROW = [
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
]

NOSE = [
    (168, 6),
    (6, 197),
    (197, 195),
    (195, 5),
    (5, 4),
    (4, 1),
    (1, 19),
    (19, 94),
    (94, 2),
]

LIPS = [
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308),
]

ALL_CONTOURS = [
    FACE_OVAL,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_BROW,
    RIGHT_BROW,
    NOSE,
    LIPS,
]


# ---------------------------------------------------------------------------
# Face drawing helpers
# ---------------------------------------------------------------------------


def draw_face_mesh(
    frame: np.ndarray,
    faces,
    show_wireframe: bool = True,
    show_labels: bool = True,
    overlay_color=config.OVERLAY_COLOR,
    font_scale: float = config.FONT_SCALE,
    font_thickness: int = config.FONT_THICKNESS,
    line_thickness: int = config.OVERLAY_THICKNESS,
    face_id_names: dict[int, str] | None = None,
) -> np.ndarray:
    """Draw face mesh and attribute labels on *frame*.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame (modified in-place for speed).
    faces : list[dict]
        Each dict has ``"landmarks"`` (478 (x, y) tuples), ``"bbox"``, and
        optional ``"age"``, ``"gender"``, ``"emotion"`` keys.
    fps : float
        Current frames-per-second to overlay.
    show_wireframe : bool
        Whether to draw the 478-point mesh wireframe.
    show_labels : bool
        Whether to draw age/gender/emotion text.
    face_id_names : dict[int, str] | None
        Optional mapping of track_id to user-assigned name.
    """
    for face in faces:
        pts = face["landmarks"]
        bbox = face.get("bbox", (0, 0, 0, 0))
        x1, y1, x2, y2 = map(int, bbox)

        if show_wireframe:
            for conn_group in ALL_CONTOURS:
                for i, j in conn_group:
                    if i < len(pts) and j < len(pts):
                        cv2.line(
                            frame,
                            pts[i],
                            pts[j],
                            overlay_color,
                            line_thickness,
                            cv2.LINE_AA,
                        )

            for i in range(0, len(pts), config.FACE_POINT_STRIDE):
                cv2.circle(
                    frame,
                    (pts[i][0], pts[i][1]),
                    2,
                    overlay_color,
                    -1,
                    lineType=cv2.LINE_AA,
                )

        if show_labels:
            parts = []
            if "age" in face:
                parts.append(f"Age: {face['age']}")
            if "gender" in face:
                parts.append(f"{face['gender']}")
            if "race" in face:
                parts.append(f"{face['race']}")
            if "emotion" in face:
                em = face["emotion"]
                parts.append(f"{em}")
            if parts:
                label = " | ".join(parts)
                cx_text = (x1 + x2) // 2
                (tw, th), _ = cv2.getTextSize(
                    label,
                    config.OVERLAY_FONT,
                    font_scale,
                    font_thickness,
                )
                tx = max(cx_text - tw // 2, 4)
                ty = max(y1 - 8, th + 4)
                cv2.putText(
                    frame,
                    label,
                    (tx, ty),
                    config.OVERLAY_FONT,
                    font_scale,
                    (0, 255, 255),
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )

        if show_labels and "spoof_confidence" in face:
            conf = face["spoof_confidence"]
            spoof_label = f"Human: {int(conf * 100)}%"
            if "track_id" in face:
                tid = face["track_id"]
                name = face_id_names.get(tid, "") if face_id_names else ""
                if name:
                    spoof_label = f"{name} | " + spoof_label
                else:
                    spoof_label = f"ID: {tid} | " + spoof_label

            cx_s = (x1 + x2) // 2
            (sw, sh), _ = cv2.getTextSize(
                spoof_label,
                config.OVERLAY_FONT,
                font_scale,
                font_thickness,
            )
            sx = max(cx_s - sw // 2, 4)
            sy = min(y2 + sh + 8, frame.shape[0] - 4)
            cv2.putText(
                frame,
                spoof_label,
                (sx, sy),
                config.OVERLAY_FONT,
                font_scale,
                (0, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA,
            )

    return frame


def apply_privacy(
    frame: np.ndarray,
    faces,
    mode: str,
    inplace: bool = False,
) -> np.ndarray:
    """Blur face regions in *frame* using the selected *mode*.

    Modes: "Pixelate", "Gaussian", "EllipticalBlur".  Pass "None" to return
    the frame unchanged.
    """
    if mode == "None" or not faces:
        return frame if inplace else frame.copy()

    bboxes = [f["bbox"] for f in faces]
    try:
        method = mode.lower()
        if method == "ellipticalblur":
            method = "elliptical"
        bf = BlurFace(method=method)
        return bf.blur_regions(frame, bboxes, inplace)
    except Exception:
        return frame if inplace else frame.copy()
