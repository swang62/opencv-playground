# OpenCV Playground

Real-time local vision playground built around [YOLOE-26](https://docs.ultralytics.com/models/yoloe/) for open-ended object/keyword search, and various facial/body recognition models for identity detection and tracking.

## Core features

- Multiple detection modes: natural language prompts, prompt-free auto-detect, and tracking features
- Multiple input sources: webcam, local video files, and YouTube links
- Persistent face + body tracking and linking with stable track IDs and inference caching

## Modes

| Mode   | Core functionality                                                             | Demo                                 |
| ------ | ------------------------------------------------------------------------------ | ------------------------------------ |
| Find   | Search for objects with prompts like `"where is my red mug"`                   | ![Find tab](public/find_tab.gif)     |
| Detect | Detect all visible objects with custom thresholds                              | ![Detect tab](public/detect_tab.gif) |
| Face   | Face mesh, body skeleton, facial attributes, image filters, and re-ID tracking | ![Face tab](public/face_tab.gif)     |

## Input sources

| Source     | Behavior                                                        |
| ---------- | --------------------------------------------------------------- |
| Webcam     | Live camera feed                                                |
| Video file | Native file picker, infinite looping, seek bar                  |
| YouTube    | Paste a URL and stream with buffered playback (requires yt-dlp) |

All modes work the same across all sources.

## Libraries used for object detection

| Component               | Used for                                                |
| ----------------------- | ------------------------------------------------------- |
| **YOLOE-26**            | Open-vocabulary object detection                        |
| **YOLO-11 + OSNet**     | Body detection and identification                       |
| **MobileCLIP**          | Text embeddings for natural-language prompts            |
| **MediaPipe**           | 478-point face/skeleton/hand meshes                     |
| **SCRFD + BYTETracker** | Face detecting and tracking                             |
| **ArcFace**             | Face recognition and identification                     |
| **UniFace**             | Facial attributes: age, gender, emotion, race, spoofing |

## Setup

Requires Python 3.13+.

Optimized for Apple Silicon, CoreMl/MPS automatically used and falls back to CPU when needed. First launch will be slowest to download all models.

```bash
uv sync
uv run python main.py   # http://127.0.0.1:8765
```

### Optional: YouTube support

```bash
brew install yt-dlp     # macOS
# or: pipx install yt-dlp
```

## Troubleshooting

| Symptom              | Likely cause                                                |
| -------------------- | ----------------------------------------------------------- |
| Camera not ready     | Camera is in use, permission was denied, or no webcam       |
| Model loading failed | First-run model download failed or corrupted                |
| No detections        | Confidence threshold is too high or environment is too dark |
| No faces             | Tracking is disabled or people/faces too small              |
| Low FPS              | Make sure no other programs are using CPU/GPU               |
| YouTube fails        | `yt-dlp` is missing, or YouTube cookies are unavailable     |

## Acknowledgements

Built on top of giants:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [MediaPipe](https://github.com/google-ai-edge/mediapipe)
- [UniFace](https://github.com/face-hh/uniface)
- [FAISS](https://github.com/facebookresearch/faiss)
- [NiceGUI](https://github.com/zauberzeug/nicegui)

## License

MIT. See [LICENSE](LICENSE).
