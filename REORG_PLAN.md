# Reorganization Plan — `src/` Flat-to-Package Refactor

## Goal

Replace the flat 14-file `src/` layout with a structured package of focused modules
grouped by concern, preserving 100% of imports, public symbols, and behavior.

## Target Architecture

```
src/
  __init__.py              # package marker (empty)
  config.py                # flat — configuration constants
  state.py                 # flat — AppState dataclass + color helpers + predict kwargs
  utils/
    __init__.py
    query.py               # normalize_query, WRAPPER_PATTERNS, FILLER_WORDS_RE
    file_picker.py         # pick_video, _pick_macos, _pick_linux, _pick_windows
  sources/
    __init__.py
    camera.py              # create_camera, NativeMacCamera, FrameReceiver, HAVE_NATIVE_CAMERA, IS_MACOS
    video_source.py        # VideoFilePlayer
    youtube_source.py      # YouTubeSource, BUFFER_SIZE, STREAM_RECONNECT_DELAY
  detection/
    __init__.py
    extraction.py          # extract_detections
    face_engine.py         # FaceEngine, download_model, LEFT_EYE_INDICES, RIGHT_EYE_INDICES, module-level paths/URLs
    body_engine.py         # BodyEngine, _download_model, module-level paths/URLs
    body_reid.py           # BodyIdEngine, _parse_tracking_results, _crop_person, _OSNET_*, _MEAN, _STD
  overlay/
    __init__.py
    annotations.py         # annotate_frame, draw_corner_bbox, draw_body_boxes, green_buf
    face_overlay.py        # draw_face_mesh, apply_privacy, ALL_CONTOURS, FACE_OVAL through LIPS
    skeleton.py            # draw_pose_skeleton, draw_hand_skeleton, POSE_LANDMARKS, POSE_CONNECTIONS,
                           #   HAND_CONNECTIONS, BODY_LANDMARK_RANGE, BODY_CONNECTIONS,
                           #   BODY_JOINT_INDICES, HAND_JOINT_INDICES
    filters.py             # apply_visual_filter, FILTER_NAMES
  pipeline/
    __init__.py
    pipeline.py            # CapturePipeline
    models.py              # ModelBundle, load_model_bundle, get_device, get_text_encoder, encode_text,
                           #   warmup_model, text_encoder
  ui/
    __init__.py
    app.py                 # index(), main(), start_services(), stop_services(), page UI + lifecycle
```

---

## Step 0 — Create directory structure

Create all new directories with `__init__.py`:

```bash
mkdir -p src/utils src/sources src/detection src/overlay src/pipeline src/ui
touch src/utils/__init__.py src/sources/__init__.py src/detection/__init__.py \
      src/overlay/__init__.py src/pipeline/__init__.py src/ui/__init__.py
```

---

## Step 1 — Whole-file moves (no split)

Pick up each file as-is and move it to its new location. No content changes besides
the `__init__.py` package marker update and import rewrites (Step 4).

| # | Old path | New path | Lines |
|---|----------|----------|-------|
| 1a | `src/file_picker.py` | `src/utils/file_picker.py` | 91 |
| 1b | `src/utils.py` | `src/utils/query.py` | 48 |
| 1c | `src/camera.py` | `src/sources/camera.py` | 158 |
| 1d | `src/video_source.py` | `src/sources/video_source.py` | 122 |
| 1e | `src/youtube_source.py` | `src/sources/youtube_source.py` | 149 |
| 1f | `src/filters.py` | `src/overlay/filters.py` | 50 |
| 1g | `src/body_reid.py` | `src/detection/body_reid.py` | 406 |
| 1h | `src/pipeline.py` | `src/pipeline/pipeline.py` | 737 |
| 1i | `src/models.py` | `src/pipeline/models.py` | 205 |
| 1j | `src/app.py` | `src/ui/app.py` | 1331 |

**Technique**: `git mv src/<old>.py src/<folder>/<new>.py`

---

## Step 2 — File splits (create new files from old content)

These files are too large or mix distinct concerns. Each old file is deleted after
all its pieces are created.

### 2a — `src/detection.py` → `src/detection/extraction.py` + `src/overlay/annotations.py`

**Source**: 191 lines.

**extraction.py** — pipeline data extraction (no drawing):

| Symbol | Kind |
|--------|------|
| `extract_detections()` | public function |
| No module-level mutable state | |

**annotations.py** — frame annotation and body-box drawing:

| Symbol | Kind |
|--------|------|
| `annotate_frame()` | public function |
| `draw_corner_bbox()` | module-private helper (used by annotate_frame + draw_body_boxes) |
| `draw_body_boxes()` | public function |
| `green_buf` | module-level mutable buffer (moves with annotation code) |

Both new files import `from src import config`.

**Old file removed**: `src/detection.py`

---

### 2b — `src/face.py` → `src/detection/face_engine.py` + `src/overlay/face_overlay.py`

**Source**: 1070 lines — the largest non-app file, mixing engine logic, drawing,
and mesh topology constants.

**face_engine.py** — FaceEngine class + model download + eye-index sets:

| Symbol | Kind |
|--------|------|
| `MODEL_DIR` | module constant |
| `MODEL_FILE` | module constant |
| `FACE_IDENTITIES_PATH` | module constant |
| `MODEL_URL` | module constant |
| `download_model()` | public function |
| `logger` | module logger |
| `LEFT_EYE_INDICES` | module constant (hardcoded set, see notes) |
| `RIGHT_EYE_INDICES` | module constant (hardcoded set, see notes) |
| `FaceEngine` | public class (all methods + attributes from original) |

The eye-index sets are hardcoded in the new file instead of being computed from
connection-pair lists. The indices are stable MediaPipe canonical values; this
avoids pulling mesh-topology data (170+ lines) into the engine file and avoids a
cross-folder dependency on overlay.

```python
# Replace current computed form:
#   LEFT_EYE_INDICES = sorted({i for pair in LEFT_EYE for i in pair})
# with hardcoded values:
LEFT_EYE_INDICES = {33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173}
RIGHT_EYE_INDICES = {263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398}
```

Imports: `from __future__ import annotations`, `concurrent.futures`, `logging`,
`ssl`, `threading`, `urllib.request`, `collections.deque`, `pathlib.Path`, `cv2`,
`mediapipe`, `numpy`, `torch`, `mediapipe.tasks.*`, `uniface.*`, `from src import config`.

**face_overlay.py** — drawing and privacy blur:

| Symbol | Kind |
|--------|------|
| `FACE_OVAL` | module constant (connection list) |
| `LEFT_EYE` | module constant (connection list) |
| `RIGHT_EYE` | module constant (connection list) |
| `LEFT_BROW` | module constant (connection list) |
| `RIGHT_BROW` | module constant (connection list) |
| `NOSE` | module constant (connection list) |
| `LIPS` | module constant (connection list) |
| `ALL_CONTOURS` | module constant (list of connection lists) |
| `draw_face_mesh()` | public function |
| `apply_privacy()` | public function |

Imports: `from __future__ import annotations`, `cv2`, `numpy`, `from src import config`,
`from uniface.privacy import BlurFace`.

**Old file removed**: `src/face.py`

---

### 2c — `src/body.py` → `src/detection/body_engine.py` + `src/overlay/skeleton.py`

**Source**: 328 lines — engine logic + skeleton topology constants + drawing.

**body_engine.py** — BodyEngine class + model download:

| Symbol | Kind |
|--------|------|
| `MODEL_DIR` | module constant |
| `POSE_URL` | module constant |
| `POSE_FILE` | module constant |
| `HAND_URL` | module constant |
| `HAND_FILE` | module constant |
| `_download_model()` | module-private function |
| `logger` | module logger |
| `BodyEngine` | public class (all methods + attributes) |

Imports: `from __future__ import annotations`, `logging`, `ssl`, `threading`,
`urllib.request`, `pathlib.Path`, `cv2`, `mediapipe`, `numpy`, `mediapipe.tasks.*`,
`from src import config`.

**skeleton.py** — pose/hand topology constants + drawing:

| Symbol | Kind |
|--------|------|
| `POSE_LANDMARKS` | module constant (dict) |
| `POSE_CONNECTIONS` | module constant (connection list) |
| `HAND_CONNECTIONS` | module constant (connection list) |
| `BODY_LANDMARK_RANGE` | module constant (set) |
| `BODY_CONNECTIONS` | module constant (filtered list) |
| `BODY_JOINT_INDICES` | module constant (set) |
| `HAND_JOINT_INDICES` | module constant (set) |
| `_download_model()` is NOT needed here | |
| `draw_pose_skeleton()` | public function |
| `draw_hand_skeleton()` | public function |

Imports: `from __future__ import annotations`, `cv2`, `numpy`, `from src import config`.

**Old file removed**: `src/body.py`

---

## Step 3 — Directories to clean up after all moves

After Steps 1–2 are complete, these top-level files are empty and should be deleted:

| File | Reason |
|------|--------|
| `src/utils.py` | content moved to `src/utils/query.py` |
| `src/file_picker.py` | content moved to `src/utils/file_picker.py` |
| `src/camera.py` | moved to `src/sources/camera.py` |
| `src/video_source.py` | moved to `src/sources/video_source.py` |
| `src/youtube_source.py` | moved to `src/sources/youtube_source.py` |
| `src/filters.py` | moved to `src/overlay/filters.py` |
| `src/detection.py` | split into `src/detection/extraction.py` + `src/overlay/annotations.py` |
| `src/face.py` | split into `src/detection/face_engine.py` + `src/overlay/face_overlay.py` |
| `src/body.py` | split into `src/detection/body_engine.py` + `src/overlay/skeleton.py` |
| `src/body_reid.py` | moved to `src/detection/body_reid.py` |
| `src/pipeline.py` | moved to `src/pipeline/pipeline.py` |
| `src/models.py` | moved to `src/pipeline/models.py` |
| `src/app.py` | moved to `src/ui/app.py` |

Remaining flat files: `src/__init__.py`, `src/config.py`, `src/state.py`.

---

## Step 4 — Import rewrites

All cross-file imports must be updated. No logic changes.

### 4a — `main.py` (entry point)

| Old import | New import |
|------------|------------|
| `from src.app import main` | `from src.ui.app import main` |

### 4b — `src/ui/app.py` (moved from `src/app.py`)

| Old import | New import |
|------------|------------|
| `from src import config` | _(unchanged — config stays flat)_ |
| `from src.file_picker import pick_video` | `from src.utils.file_picker import pick_video` |
| `from src.models import ModelBundle, get_device, load_model_bundle` | `from src.pipeline.models import ModelBundle, get_device, load_model_bundle` |
| `from src.pipeline import CapturePipeline` | `from src.pipeline.pipeline import CapturePipeline` |
| `from src.state import COLOR_MAP, AppState, color_name_to_hex` | _(unchanged — state stays flat)_ |
| `from src.utils import normalize_query` | `from src.utils.query import normalize_query` |
| `from src.video_source import VideoFilePlayer` | `from src.sources.video_source import VideoFilePlayer` |
| `from src.youtube_source import YouTubeSource` | `from src.sources.youtube_source import YouTubeSource` |

### 4c — `src/pipeline/pipeline.py` (moved from `src/pipeline.py`)

| Old import | New import |
|------------|------------|
| `from src import config` | _(unchanged)_ |
| `from src.body import draw_hand_skeleton, draw_pose_skeleton` | `from src.overlay.skeleton import draw_hand_skeleton, draw_pose_skeleton` |
| `from src.camera import create_camera` | `from src.sources.camera import create_camera` |
| `from src.detection import annotate_frame, draw_body_boxes, extract_detections` | `from src.detection.extraction import extract_detections` |
| | `from src.overlay.annotations import annotate_frame, draw_body_boxes` |
| `from src.face import apply_privacy, draw_face_mesh` | `from src.overlay.face_overlay import apply_privacy, draw_face_mesh` |
| `from src.filters import apply_visual_filter` | `from src.overlay.filters import apply_visual_filter` |
| `from src.state import color_name_to_bgr, get_predict_kwargs` | _(unchanged)_ |

### 4d — `src/pipeline/models.py` (moved from `src/models.py`)

| Old import | New import |
|------------|------------|
| `from src import config` | _(unchanged)_ |
| `from src.body import BodyEngine` | `from src.detection.body_engine import BodyEngine` |
| `from src.body_reid import BodyIdEngine` | `from src.detection.body_reid import BodyIdEngine` |
| `from src.face import FaceEngine` | `from src.detection.face_engine import FaceEngine` |

### 4e — Files that move as packages (internal relative imports)

These files only import `from src import config` (which stays flat) or have no
internal imports, so they need no import changes after the move:

| File | Internal imports |
|------|-----------------|
| `src/utils/query.py` | none |
| `src/utils/file_picker.py` | none |
| `src/sources/camera.py` | `from src import config` → stays |
| `src/sources/video_source.py` | none |
| `src/sources/youtube_source.py` | none |
| `src/overlay/filters.py` | none |
| `src/overlay/annotations.py` | `from src import config` → stays |
| `src/overlay/skeleton.py` | `from src import config` → stays |
| `src/overlay/face_overlay.py` | `from src import config` → stays |
| `src/detection/extraction.py` | `from src import config` → stays |
| `src/detection/body_engine.py` | `from src import config` → stays |
| `src/detection/body_reid.py` | `from src import config` → stays |
| `src/detection/face_engine.py` | `from src import config` → stays |

---

## Dependency / Risk Notes

### RISK 1: `src/ui/app.py` (1331 lines, former `src/app.py`)
**Why risky**: Module-level globals (`state`, `pipeline`, `bundle`, `device_str`,
`_current_video_source`) shared across UI callbacks and lifecycle hooks. The `index()`
function is a single 1241-line closure that defines nested functions referencing
`nonlocal` variables and closures captured from UI components. This is the highest-risk
move because every import path converges here.
**Mitigation**: Move whole, split nothing. The one import change (`from src.app`
→ `from src.ui.app`) in `main.py` is mechanical and trivially testable.
**Future**: Consider splitting `index()` into page components after the move settles.

### RISK 2: `src/pipeline/pipeline.py` (737 lines, former `src/pipeline.py`)
**Why risky**: Import list changes are the most complex (5 module paths change),
and the inference loop threads interact with UI state. An import error here breaks
all detection.
**Mitigation**: Each changed import has a direct 1:1 mapping shown above. Verify
with `python -c "from src.pipeline.pipeline import CapturePipeline"` after changes.

### RISK 3: File splits — `face.py` and `body.py`
**Why risky**: Splitting a file introduces the possibility of missing a symbol.
The eye-index sets must be manually hardcoded in `face_engine.py` instead of
computed from connection lists.
**Mitigation**:
- Every public symbol from the old files is explicitly accounted for in the tables above.
- The hardcoded index sets are verified against the original `LEFT_EYE` and `RIGHT_EYE`
  connection lists at the time of the plan. They must match precisely.
- After split, run `python -c "from src.detection.face_engine import FaceEngine; from src.overlay.face_overlay import draw_face_mesh"` etc. as a smoke test.

### RISK 4: `_download_model()` duplication
In the current `body.py`, there is a single `_download_model()` function used by
both `BodyEngine.ensure_pose_loaded()` and `BodyEngine.ensure_hand_loaded()`.
After split, this function stays with `BodyEngine` in `detection/body_engine.py`.
The overlay `skeleton.py` does NOT need it. No duplication.

### RISK 5: Naming collision — `src.detection` and `src.detection.extraction`
The old `src/detection.py` had a top-level module `detection`. Its `extract_detections`
function was imported as `from src.detection import extract_detections`. After the split,
it becomes `from src.detection.extraction import extract_detections`. Every existing
caller (only `pipeline.py`) must be updated — covered in Step 4c.

---

## Execution Order Summary

| Phase | Step | What | Risk |
|-------|------|------|------|
| 0 | Create directories | `mkdir + touch __init__.py` | none |
| 1a–1j | Move whole files | `git mv` each old→new | low |
| 2a | Split `detection.py` | create 2 files, delete old | low |
| 2b | Split `face.py` | create 2 files, delete old (hardcode indices) | **medium** |
| 2c | Split `body.py` | create 2 files, delete old | low |
| 3 | Delete empty old files | `git rm` after all moves/splits done | none |
| 4a | Update `main.py` | 1 import change | low |
| 4b | Update `src/ui/app.py` | 8 import changes | low (direct 1:1) |
| 4c | Update `src/pipeline/pipeline.py` | **6 import changes, 2 new lines** | **medium** |
| 4d | Update `src/pipeline/models.py` | 3 import changes | low |
| Verify | Smoke-test | `python -c "from src.ui.app import main"` | — |

Total new files created: 18 (6 `__init__.py` + 12 module files)
Total old files deleted: 13 (all flat files except `config.py`, `state.py`, `__init__.py`)
Total import lines changed: ~19 lines

---

## Post-Move Size Comparison

| File | Before (lines) | After (lines) |
|------|------:|------:|
| `app.py` / `ui/app.py` | 1331 | 1331 |
| `pipeline.py` / `pipeline/pipeline.py` | 737 | 737 |
| `face.py` → `detection/face_engine.py` + `overlay/face_overlay.py` | 1070 | ~660 + ~150 = ~810 |
| `body_reid.py` / `detection/body_reid.py` | 406 | 406 |
| `body.py` → `detection/body_engine.py` + `overlay/skeleton.py` | 328 | ~140 + ~190 = ~330 |
| `models.py` / `pipeline/models.py` | 205 | 205 |
| `detection.py` → `detection/extraction.py` + `overlay/annotations.py` | 191 | ~55 + ~140 = ~195 |
| `camera.py` / `sources/camera.py` | 158 | 158 |
| `youtube_source.py` / `sources/youtube_source.py` | 149 | 149 |
| `video_source.py` / `sources/video_source.py` | 122 | 122 |
| `file_picker.py` / `utils/file_picker.py` | 91 | 91 |
| `config.py` | 70 | 70 (flat) |
| `filters.py` / `overlay/filters.py` | 50 | 50 |
| `utils.py` / `utils/query.py` | 48 | 48 |
| `state.py` | 166 | 166 (flat) |
| **Total** | **~5122** | **~5122** |

Line counts stay identical (or near-identical for splits) — no logic changes.

---

## Verification Checklist

After executing all steps, run these checks:

```bash
# 1. No old flat files remain (except config.py, state.py, __init__.py)
ls src/*.py
# Expect: __init__.py  config.py  state.py

# 2. New structure exists
ls src/utils/ src/sources/ src/detection/ src/overlay/ src/pipeline/ src/ui/

# 3. Smoke-test imports
python -c "from src.ui.app import main"
python -c "from src.pipeline.pipeline import CapturePipeline"
python -c "from src.pipeline.models import ModelBundle"
python -c "from src.detection.face_engine import FaceEngine, download_model"
python -c "from src.detection.body_engine import BodyEngine"
python -c "from src.detection.body_reid import BodyIdEngine"
python -c "from src.detection.extraction import extract_detections"
python -c "from src.overlay.annotations import annotate_frame, draw_body_boxes"
python -c "from src.overlay.face_overlay import draw_face_mesh, apply_privacy"
python -c "from src.overlay.skeleton import draw_pose_skeleton, draw_hand_skeleton"
python -c "from src.overlay.filters import apply_visual_filter"
python -c "from src.sources.camera import create_camera"
python -c "from src.sources.video_source import VideoFilePlayer"
python -c "from src.sources.youtube_source import YouTubeSource"
python -c "from src.utils.query import normalize_query"
python -c "from src.utils.file_picker import pick_video"

# 4. Run the app briefly to verify no import errors at startup
timeout 5 uv run python main.py 2>&1 | head -20
```
