# OpenCV Playground

Local webcam object detection with [YOLOE-26](https://docs.ultralytics.com/models/yoloe/)
open-ended prompting, served by [NiceGUI](https://nicegui.io/).

## Setup

Requires Python >= 3.13. Apple Silicon (MPS) is used automatically; falls back to CPU if unavailable. On first launch, all models will need to be downloaded first.

```bash
uv sync
uv run python main.py # http://127.0.0.1:8080
```

## Modes

- **Find** — Enter a natural-language query to search for anything in the webcam feed. Ideally search for things like **adjective** + **noun** (e.g. "red apple").
- **Everything** — Uses the prompt-free model to detect all objects in the webcam feed. You can control the top K items and confidence threshold.

## Troubleshooting

| Symptom                       | Likely cause                                                                                           |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| "Camera not ready"            | Camera in use by another app, denied permission, or no webcam connected.                               |
| "Model loading failed"        | Network issue on first run. Check internet, re-launch.                                                 |
| No detections                 | Confidence too high; lower the slider.                                                                 |
| Low FPS                       | MPS fallback to CPU. Close other GPU-intensive apps.                                                   |
| "No objects detected"         | Mode is correct but nothing matches the target. Show **Everything** to verify the camera sees objects. |
| Ctrl-C doesn't release camera | Force-quit the terminal or Activity Monitor. The shutdown hook usually handles this.                   |
