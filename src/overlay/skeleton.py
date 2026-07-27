"""Skeleton topology constants and drawing helpers for pose + hand landmarks."""

from __future__ import annotations

import cv2
import numpy as np

from src import config

# ---------------------------------------------------------------------------
# Pose landmark indices (MediaPipe 33-point)
# ---------------------------------------------------------------------------

POSE_LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (27, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),
]

# Hand landmark indices (MediaPipe 21-point)
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (1, 5),
    (5, 9),
    (9, 13),
    (13, 17),
]

# ---------------------------------------------------------------------------
# Drawing helpers — body skeleton filtered to torso + limbs only
# ---------------------------------------------------------------------------

# Head (0-10), finger bases (17-22), and feet (29-32) are excluded.
BODY_LANDMARK_RANGE = set(range(11, 17)) | set(range(23, 29))
BODY_CONNECTIONS = [
    (i, j)
    for i, j in POSE_CONNECTIONS
    if i in BODY_LANDMARK_RANGE and j in BODY_LANDMARK_RANGE
]
BODY_JOINT_INDICES = BODY_LANDMARK_RANGE


def draw_pose_skeleton(
    frame: np.ndarray, landmarks, color=None, thickness=None, joint_radius=None
):
    if color is None:
        color = config.SKELETON_COLOR
    if thickness is None:
        thickness = config.SKELETON_THICKNESS
    if joint_radius is None:
        joint_radius = config.JOINT_RADIUS
    for i, j in BODY_CONNECTIONS:
        cv2.line(frame, landmarks[i], landmarks[j], color, thickness, cv2.LINE_AA)
    for i in BODY_JOINT_INDICES:
        if i < len(landmarks):
            cv2.circle(frame, landmarks[i], joint_radius, color, -1, cv2.LINE_AA)


# Hand: wrist(0), MCP knuckles(1,5,9,13,17), and fingertips(4,8,12,16,20).
HAND_JOINT_INDICES = {0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20}


def draw_hand_skeleton(
    frame: np.ndarray, landmarks, color=None, thickness=None, joint_radius=None
):
    if color is None:
        color = config.SKELETON_COLOR
    if thickness is None:
        thickness = config.SKELETON_THICKNESS
    if joint_radius is None:
        joint_radius = config.JOINT_RADIUS
    for i, j in HAND_CONNECTIONS:
        if i < len(landmarks) and j < len(landmarks):
            cv2.line(frame, landmarks[i], landmarks[j], color, thickness, cv2.LINE_AA)
    for i in HAND_JOINT_INDICES:
        if i < len(landmarks):
            cv2.circle(frame, landmarks[i], joint_radius, color, -1, cv2.LINE_AA)
