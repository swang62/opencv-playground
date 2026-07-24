"""Application configuration — all user-facing constants in one place."""

# Model paths (relative to project root)
PROMPTED_MODEL = "models/yoloe-26s-seg.pt"
PROMPTFREE_MODEL = "models/yoloe-26s-seg-pf.pt"

# Inference size (pixels, squared). Lower = faster, less sensitive to small objects.
INFERENCE_SIZE = 480

# Camera
CAMERA_INDEX = 0

# Server
HOST = "127.0.0.1"
PORT = 8765
TITLE = "Object Finder"

# UI defaults
DEFAULT_CONFIDENCE = 0.2
CONFIDENCE_MIN = 0.05
CONFIDENCE_MAX = 0.95
CONFIDENCE_STEP = 0.05

# Overlay font sizes (OpenCV FONT_HERSHEY_SIMPLEX scale & thickness)
LABEL_FONT_SCALE = 1.0
LABEL_FONT_THICKNESS = 2
FPS_FONT_SCALE = 1.0
FPS_FONT_THICKNESS = 1
