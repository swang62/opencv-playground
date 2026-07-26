"""Application configuration — all user-facing constants in one place."""

import cv2

# ── Paths (relative to project root) ──────────────────────────────────────────
MODELS_DIR = "models"
BODY_IDENTITIES_DIR = f"{MODELS_DIR}/body-identities"
BODY_THUMBNAILS_DIR = f"{MODELS_DIR}/body-thumbnails"
PROMPTED_MODEL = f"{MODELS_DIR}/yoloe-26l-seg.pt"
PROMPTFREE_MODEL = f"{MODELS_DIR}/yoloe-26l-seg-pf.pt"
BODY_DETECTION_MODEL = "yolo11s.pt"
DETECT_CONFIDENCE = 0.45
BODY_REID_MODEL = f"{MODELS_DIR}/osnet_x1_0_msmt17.onnx"

# ── Inference size ────────────────────────────────────────────────────────────
INFERENCE_SIZE = 640
INFERENCE_UPDATE_INTERVAL = 20
DETECT_UPDATE_INTERVAL = 5

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
MAX_DETECT_BOX_AREA_RATIO = 0.4  # max bbox area ratio of frame to attempt detection

# ── Server ────────────────────────────────────────────────────────────────────
HOST = "127.0.0.1"
PORT = 8765
TITLE = "Real-Time Object Detection"

# ── Identity thresholds (shared by face and body Re-ID) ───────────────────────
IDENTITY_SIMILARITY_THRESHOLD = 0.6
IDENTITY_REMOVAL_FRAMES = 5  # consecutive empty frames before clearing
IDENTITY_MIN_AREA_RATIO = 0.1  # min bbox area ratio of frame for re-id
IDENTITY_CHIP_WIDTH = 88
IDENTITY_CHIP_HEIGHT = 112
IDENTITY_THUMBNAIL_SIZE = 64

# ── Overlay appearance ────────────────────────────────────────────────────────
OVERLAY_COLOR = (0, 255, 0)
OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_THICKNESS = 5
FONT_SCALE = 1.5
FONT_THICKNESS = 2
ALPHA = 0.3

# ── UI defaults ───────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.2
FIND_CONFIDENCE = 0.1
CONFIDENCE_MIN = 0.05
CONFIDENCE_MAX = 0.95
CONFIDENCE_STEP = 0.05

# ── Page layout ───────────────────────────────────────────────────────────────
PAGE_PADDING_VERTICAL = 16
PAGE_PADDING_HORIZONTAL = 12

# ── Re-ID embedding buffer ────────────────────────────────────────────────────
REID_EMBEDDING_BUFFER_SIZE = 5  # frames averaged per track before matching/enrolling

# ── Age smoothing ─────────────────────────────────────────────────────────────
AGE_SMOOTHING_ALPHA = 0.1  # EMA coefficient for temporal age smoothing

# ── Body / hand skeleton ──────────────────────────────────────────────────────
SKELETON_COLOR = OVERLAY_COLOR
SKELETON_THICKNESS = 4
JOINT_RADIUS = 6
FACE_POINT_STRIDE = 4
FACE_DETECTION_INPUT_SIZE = (INFERENCE_SIZE, INFERENCE_SIZE)
