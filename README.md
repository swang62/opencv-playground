# OpenCV Playground

Real-time local vision playground built with [YOLOE-26](https://docs.ultralytics.com/models/yoloe/) and [NiceGUI](https://nicegui.io/).

Search for objects in plain English, detect everything in frame, or switch to face mode for face analysis and identity tracking.

## At a glance

* **3 modes** — Find, Detect, Face
* **3 sources** — webcam, local video, YouTube
* **Persistent identity tracking** — face + body re-ID across frames and restarts
* **Playback controls** — play, pause, seek, zoom, source switching

## Modes

### Find

* Type queries like `"where is my red mug"`
* Use quick category pills for common targets
* Shows the active search target

### Detect

* Detect all visible objects
* **Top-K control** limits results to the most relevant 1 to 10 detections
* **Confidence threshold** filters weaker detections
* Temporal label smoothing reduces flicker

### Face

* **Face mesh** overlay
* **Face attributes** including age, gender, emotion, race, and liveness
* **Privacy mode**: none, pixelate, or gaussian blur
* **Image filters**: heat, CRT, comic, invert
* **Tracking toggle** enables face + body re-identification
* **Body mesh** adds pose and hand overlays

## Sources

### Webcam

* Live camera input
* Starts on demand

### Video file

* Open any local video from the native file picker
* Infinite looping playback
* Seek bar with timestamp
* Pause and resume without losing the current frame

### YouTube

* Paste a YouTube URL and play it in the same interface
* Buffered playback with reconnect support

All modes work the same across all sources.

## Tracking and identity

* **Face re-ID** recognizes previously saved faces
* **Body re-ID** recognizes previously saved bodies
* **Track IDs** stay attached as people move through frame
* **Buffered matching** waits for enough evidence before naming a track, which cuts down flicker
* **Face-body auto-linking** connects a face track to a body track when they belong to the same person
* **Shared naming** lets face and body identities inherit each other's name
* **Live identity panels** show tracked faces and bodies with thumbnails and add/remove naming actions

## Controls

* **Play / stop** for webcam
* **Pause / resume** for video and YouTube
* **Seek bar** for local video
* **Drag-to-zoom ROI** with reset
* **Source switching** between webcam, video file, and YouTube
* **Overlay controls** for color, font scale, and line thickness

## Setup

Requires Python 3.13+.

Apple Silicon uses MPS automatically and falls back to CPU when needed. Models download on first launch.

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

| Symptom | Likely cause |
|---|---|
| "Camera not ready" | Camera is in use, permission was denied, or no webcam is connected |
| "Model loading failed" | First-run model download failed; check network and restart |
| No detections | Confidence threshold is too high |
| Low FPS | Running on CPU instead of MPS |
| "No objects detected" | Nothing matches the Find query; switch to Detect to verify the feed |
| YouTube fails | `yt-dlp` is missing, or YouTube cookies are unavailable |
| Ctrl-C doesn't release camera | Force-quit the process if shutdown did not complete cleanly |
