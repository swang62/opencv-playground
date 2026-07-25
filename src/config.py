"""Application configuration — all user-facing constants in one place."""

import cv2

# Model paths (relative to project root)
MODELS_DIR = "models"
PROMPTED_MODEL = f"{MODELS_DIR}/yoloe-26l-seg.pt"
PROMPTFREE_MODEL = f"{MODELS_DIR}/yoloe-26l-seg-pf.pt"

# Inference size (pixels, squared). Lower = faster, less sensitive to small objects.
INFERENCE_SIZE = 640

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

CAMERA_UPDATE_INTERVAL = 20
MAX_DETECT_BOX_AREA_RATIO = 0.5

# Server
HOST = "127.0.0.1"
PORT = 8765
TITLE = "Real-Time Object Detection"

# UI defaults
DEFAULT_THRESHOLD = 0.2
DEFAULT_OPACITY = 0.2

FIND_CONFIDENCE = 0.1
CONFIDENCE_MIN = 0.05
CONFIDENCE_MAX = 0.95
CONFIDENCE_STEP = 0.05

# Overlay appearance — single green used everywhere.
OVERLAY_COLOR = (0, 255, 0)
OVERLAY_THICKNESS = 4

# Overlay font
OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.5
FONT_THICKNESS = 2

# Face mesh rendering
FACE_ID_SIMILARITY_THRESHOLD = 0.5
FACE_DETECTION_INPUT_SIZE = (640, 640)

# Page layout
PAGE_MAX_WIDTH = 1200
PAGE_PADDING_VERTICAL = 24
PAGE_PADDING_HORIZONTAL = 20

# Face ID chips
FACE_CHIP_WIDTH = 88
FACE_CHIP_HEIGHT = 112
FACE_THUMBNAIL_SIZE = 64

# Body / hand skeleton
SKELETON_COLOR = OVERLAY_COLOR
SKELETON_THICKNESS = 4
JOINT_RADIUS = 6
FACE_POINT_STRIDE = 4
