"""NiceGUI web interface and application lifecycle."""

from __future__ import annotations

import asyncio
import base64
import colorsys
import hashlib
import logging
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from nicegui import app as napp
from nicegui import ui

from src import config
from src.file_picker import pick_video
from src.models import ModelBundle, get_device, load_model_bundle
from src.pipeline import CapturePipeline
from src.state import COLOR_MAP, AppState, color_name_to_hex
from src.utils import normalize_query
from src.video_source import VideoFilePlayer
from src.youtube_source import YouTubeSource

logger = logging.getLogger(__name__)

# Module-level references shared between UI and lifecycle hooks.
state: AppState = AppState()
pipeline: CapturePipeline | None = None
bundle: ModelBundle | None = None
device_str: str = "unknown"
_current_video_source: VideoFilePlayer | YouTubeSource | None = None
_session_count: int = 0
THUMBNAILS_DIR = Path(config.MODELS_DIR) / "thumbnails"
BODY_THUMBNAILS_DIR = Path(config.BODY_THUMBNAILS_DIR)


def _face_thumbnail_url(name: str) -> str | None:
    """Return a data URL for the face thumbnail, or None if missing."""
    path = THUMBNAILS_DIR / f"{name}.jpg"
    if not path.exists():
        return None
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _body_thumbnail_url(name: str) -> str | None:
    """Return a data URL for the body thumbnail, or None if missing."""
    path = BODY_THUMBNAILS_DIR / f"{name}.jpg"
    if not path.exists():
        return None
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _shared_identity_for_link(face_tid: int, body_tid: int) -> str:
    """Deterministic single identity name for a linked face-body pair.

    Returns the same value regardless of which side calls it, so that both
    the face and body chip of a linked pair use the same color key.
    When both names exist and differ, prefers the body name.
    """
    face_name = state.face_id_names.get(face_tid, "")
    body_name = state.body_id_names.get(body_tid, "")
    return body_name or face_name


def _chip_link_color(body_track_id: int, identity_name: str) -> str:
    """Deterministic hue-spread hex color for a linked pair's chip border.

    Keyed by ``body_track_id`` + ``identity_name`` so that both the face and
    body chip of a linked pair share the same outline color.
    """
    key = f"{body_track_id}:{identity_name}"
    digest = hashlib.md5(key.encode(), usedforsecurity=False).digest()
    hue = (int.from_bytes(digest[:4], "big") % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.65)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


@ui.page("/")
def index():
    """Assemble the full page."""
    ui.dark_mode().enable()

    FW = "w-full"
    IWN = "items-center w-full no-wrap"
    CAP = "text-caption"
    GROW = "flex-grow"
    current_search_status = ""
    current_frame_jpeg: bytes | None = None
    current_face_chip_key: object = ()
    current_body_chip_key: object = ()

    def set_search_status(text: str):
        nonlocal current_search_status
        if text == current_search_status:
            return
        current_search_status = text
        search_status.set_text(text)
        search_status_row.visible = bool(text)

    # Core webcam lifecycle (defined early so all controls can reference them)
    def _start():
        global pipeline, bundle, _current_video_source
        if bundle is None:
            ui.notify("Models not loaded", type="warning")
            return
        if pipeline is not None:
            return
        _current_video_source = None
        pipeline = CapturePipeline(bundle, state)
        pipeline.start()

    def _stop():
        global pipeline, _current_video_source
        _current_video_source = None
        if pipeline is not None:
            pipeline.stop()
            pipeline = None

    def _clear_display():
        nonlocal current_frame_jpeg
        current_frame_jpeg = None
        w, h = config.CAMERA_WIDTH, config.CAMERA_HEIGHT
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            blank,
            "No video stream",
            (w // 2 - 200, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (80, 80, 80),
            2,
        )
        _, jpeg = cv2.imencode(".jpg", blank)
        encoded = base64.b64encode(jpeg).decode("ascii")
        cam.set_source(f"data:image/jpeg;base64,{encoded}")

    def _start_with_video(video_path: str):
        global pipeline, _current_video_source
        _stop()
        _current_video_source = VideoFilePlayer(
            video_path, target_size=(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        )
        pipeline = CapturePipeline(
            bundle, state, camera=_current_video_source, mirror=False
        )
        pipeline.start()
        # Wait for first annotated frame so the display switches instantly
        for _ in range(200):
            if pipeline.get_latest_encoded_frame() is not None:
                break
            time.sleep(0.01)
        update_camera_toggle_button()

    async def _pick_and_play():
        path = await asyncio.to_thread(pick_video)
        if path is None:
            return
        try:
            _start_with_video(path)
        except RuntimeError as exc:
            ui.notify(f"Failed to open video: {exc}", type="negative")
            return
        filename = Path(path).name
        video_label.text = filename
        video_label.visible = True
        webcam_btn.visible = True
        youtube_btn.visible = False
        open_video_btn.visible = False
        seek_row.visible = True

    async def _pick_youtube():
        with youtube_dialog:
            youtube_url_input.value = ""
            youtube_url_input.run_method("focus")
        youtube_dialog.open()

    async def _on_youtube_play():
        url = (youtube_url_input.value or "").strip()
        if not url:
            ui.notify("Enter a YouTube URL", type="warning")
            return
        youtube_dialog.close()
        youtube_loading.visible = True
        try:
            source = await asyncio.to_thread(
                lambda: YouTubeSource(
                    url, target_size=(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
                )
            )
        except RuntimeError as exc:
            youtube_loading.visible = False
            ui.notify(str(exc), type="negative")
            return
        youtube_loading.visible = False
        global pipeline, _current_video_source
        _stop()
        _current_video_source = source
        pipeline = CapturePipeline(
            bundle, state, camera=_current_video_source, mirror=False
        )
        pipeline.start()
        for _ in range(200):
            if pipeline.get_latest_encoded_frame() is not None:
                break
            await asyncio.sleep(0.01)
        update_camera_toggle_button()
        youtube_btn.visible = False
        open_video_btn.visible = False
        webcam_btn.visible = True
        video_label.text = url
        video_label.visible = True

    # -- YouTube URL dialog --
    youtube_dialog = ui.dialog()
    with youtube_dialog, ui.card().classes("w-96"):
        ui.label("Play YouTube Video").classes("text-bold text-h6 q-mb-sm")
        youtube_url_input = ui.input(
            label="YouTube URL",
            placeholder="https://youtube.com/watch?v=...",
        ).classes("w-full")
        with ui.row().classes("w-full justify-end q-mt-sm"):
            ui.button("Cancel", on_click=youtube_dialog.close).props("flat")
            ui.button("Play", on_click=_on_youtube_play).props("color=primary")

    def _switch_to_webcam():
        global _current_video_source
        _current_video_source = None
        _stop()
        open_video_btn.visible = True
        youtube_btn.visible = True
        webcam_btn.visible = False
        video_label.visible = False
        seek_row.visible = False
        update_camera_toggle_button()
        _start()

    # ---- page chrome --------------------------------------------------------
    ui.add_head_html(f"""
    <style>
      body {{ background: #366576; }}
      .q-page {{ background: linear-gradient(180deg, #1e4356 0%, #1e4356 100%); }}
      .q-card {{ border-radius: 12px; box-shadow: 0 2px 16px rgba(0,0,0,0.35) !important; }}
      .text-h4 {{ letter-spacing: -0.02em; }}
      .control-card {{
        background: rgba(18, 29, 35, 0.72) !important;
        border: 1px solid rgba(255, 255, 255, 0.09);
        box-shadow: 0 18px 40px rgba(9, 15, 18, 0.22) !important;
        backdrop-filter: blur(14px);
      }}
      .mode-tabs {{
        padding: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }}
      .mode-tabs .q-tabs__content {{ gap: 8px; }}
      .mode-tabs .q-tab {{
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        flex-shrink: 1;
        min-width: 0;
        min-height: 46px;
        border-radius: 14px;
        color: rgba(236, 244, 247, 0.72);
        transition: background 0.18s ease, color 0.18s ease, transform 0.18s ease;
      }}
      .mode-tabs .q-tab:hover {{
        background: rgba(255, 255, 255, 0.06);
        color: rgba(255, 255, 255, 0.96);
      }}
      .mode-tabs .q-tab--active {{
        background: rgba(255, 255, 255, 0.12);
        color: rgba(255, 255, 255, 0.98);
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08), 0 10px 20px rgba(0, 0, 0, 0.12);
      }}
      .mode-tabs .q-tab__indicator {{ display: none; }}
      .mode-tabs .q-tab .q-tab__content {{ flex-direction: row; gap: 8px; padding: 6px 0; }}
      .mode-tabs .q-tab .q-tab__icon {{ margin: 0; font-size: 1.35rem; }}
      .clean-panels, .clean-panels .q-tab-panels, .clean-panels .q-panel {{ background: transparent !important; }}
      .clean-panels .q-tab-panel {{ padding: 16px 16px 14px; }}
      .premium-button {{
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.12) !important;
        color: #f7fbfc !important;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.16);
      }}
      .premium-button:hover {{ background: rgba(255, 255, 255, 0.18) !important; }}
      .subtle-button {{
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.05) !important;
        color: rgba(245, 250, 251, 0.92) !important;
        border: 1px solid rgba(255, 255, 255, 0.08);
      }}
      .subtle-button:hover {{ background: rgba(255, 255, 255, 0.08) !important; }}
      .camera-toolbar {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 6px;
        border-radius: 9999px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(20, 20, 28, 0.96);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
        overflow: hidden;
      }}
      .camera-toolbar .q-btn {{ border-radius: 9999px; min-width: 42px; min-height: 42px; }}
      .camera-toolbar .q-icon {{ font-size: 1.4rem; }}
      .camera-toolbar .q-btn--flat {{ color: #fff; }}
      .camera-toolbar .q-btn--flat:hover {{ background: rgba(255, 255, 255, 0.08); }}
      .camera-toolbar .q-btn--flat.q-btn--actionable.q-hoverable .q-focus-helper {{ display: none; }}
      .identity-chip-img {{
        width: 100% !important;
        height: {config.IDENTITY_CHIP_WIDTH}px !important;
        object-fit: cover;
        display: block;
        flex-shrink: 0;
      }}
      .identity-chip-action-btn {{
        opacity: 0;
        transition: opacity 0.15s;
        background: rgba(10, 10, 14, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #fff !important;
      }}
      .identity-chip-container:hover .identity-chip-action-btn {{ opacity: 1; }}
      .identity-chip-placeholder {{ width: 100%; height: {config.IDENTITY_CHIP_WIDTH}px; background: #fff; flex-shrink: 0; }}
      .identity-panel {{
        flex: 1 1 0;
        min-width: 260px;
        background: rgba(18, 29, 35, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 12px;
        padding: 8px;
      }}
      .identity-panel-title {{
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: rgba(236, 244, 247, 0.6);
        margin-bottom: 6px;
        padding-left: 4px;
      }}
      .identity-empty-state {{
        font-size: 0.8rem;
        color: rgba(236, 244, 247, 0.35);
        padding: 6px 4px;
      }}

      @media (max-width: 800px) {{
        .app-layout-row > div {{
          width: 100% !important;
          min-width: 100% !important;
          flex-basis: 100% !important;
        }}
      }}
    </style>
    """)

    with (
        ui.element("div")
        .classes(FW)
        .style(
            f"margin: 0 auto; padding: {config.PAGE_PADDING_VERTICAL}px {config.PAGE_PADDING_HORIZONTAL}px;"
        )
    ):
        with ui.row().classes("w-full flex-wrap items-start app-layout-row"):
            with ui.column().style("flex: 3 1 0; min-width: 0; overflow: hidden;"):
                ui.label(config.TITLE).classes(
                    "text-h4 text-weight-bold text-center w-full q-mb-md"
                )
                cam = (
                    ui.interactive_image(
                        sanitize=False,
                        events=["mousedown", "mousemove", "mouseup"],
                    )
                    .classes("w-full border-1 rounded")
                    .style("user-select: none; touch-action: none;")
                )

                # -- Control bar (under webcam, pill-shaped, like a video player) --
                roi_selection_active = False
                roi_dragging = False
                roi_drag_x1 = 0.0
                roi_drag_y1 = 0.0
                roi_overlay = cam.add_layer()

                def update_roi_overlay():
                    roi_overlay.content = ""

                def on_cam_mouse(e):
                    nonlocal \
                        roi_selection_active, \
                        roi_dragging, \
                        roi_drag_x1, \
                        roi_drag_y1
                    if not roi_selection_active:
                        return
                    if e.type == "mousedown" and e.button == 0:
                        roi_dragging = True
                        roi_drag_x1 = e.image_x
                        roi_drag_y1 = e.image_y
                        # Clear previous ROI while dragging
                        state.clear_roi()
                        update_roi_overlay()
                    elif e.type == "mousemove" and roi_dragging:
                        x1 = min(roi_drag_x1, e.image_x)
                        y1 = min(roi_drag_y1, e.image_y)
                        x2 = max(roi_drag_x1, e.image_x)
                        y2 = max(roi_drag_y1, e.image_y)
                        roi_overlay.content = (
                            f'<rect x="{x1}" y="{y1}" '
                            f'width="{x2 - x1}" height="{y2 - y1}" '
                            f'fill="{color_name_to_hex(state.overlay_color_name)}33" '
                            f'stroke="{color_name_to_hex(state.overlay_color_name)}" stroke-width="2" '
                            f'stroke-dasharray="6,3" />'
                        )
                    elif e.type == "mouseup" and roi_dragging:
                        roi_dragging = False
                        x1 = min(roi_drag_x1, e.image_x)
                        y1 = min(roi_drag_y1, e.image_y)
                        x2 = max(roi_drag_x1, e.image_x)
                        y2 = max(roi_drag_y1, e.image_y)
                        if (x2 - x1) >= 16 and (y2 - y1) >= 16:
                            state.set_roi(x1, y1, x2, y2)
                        else:
                            state.clear_roi()
                        roi_overlay.content = ""

                cam.on_mouse(on_cam_mouse)

                def on_roi_zoom():
                    nonlocal roi_selection_active
                    if roi_selection_active:
                        roi_selection_active = False
                        roi_zoom_btn.props("flat round dense color=white size=sm")
                        cam.style(
                            "user-select: none; touch-action: none; cursor: default;"
                        )
                    else:
                        roi_selection_active = True
                        roi_zoom_btn.props("flat round dense color=amber size=sm")
                        cam.style(
                            "user-select: none; touch-action: none; cursor: crosshair; cursor: cell;"
                        )

                def on_reset_zoom():
                    nonlocal roi_selection_active
                    roi_selection_active = False
                    roi_zoom_btn.props("flat round dense color=white size=sm")
                    cam.style("user-select: none; touch-action: none; cursor: default;")
                    state.clear_roi()
                    update_roi_overlay()

                def on_toggle_camera():
                    if _current_video_source is not None:
                        if pipeline is not None and pipeline.is_paused:
                            pipeline.resume()
                        elif pipeline is not None:
                            pipeline.pause()
                        update_camera_toggle_button()
                    elif pipeline is None:
                        _start()
                    else:
                        _stop()

                def update_camera_toggle_button():
                    if pipeline is None:
                        camera_toggle_button.props(
                            "flat round dense size=sm color=positive icon=play_arrow"
                        )
                    elif pipeline.is_paused:
                        camera_toggle_button.props(
                            "flat round dense size=sm color=positive icon=play_arrow"
                        )
                        camera_toggle_button.tooltip("Resume Video")
                    else:
                        camera_toggle_button.props(
                            "flat round dense size=sm color=negative icon=stop"
                        )

                # -- Video seek bar (visible only in video mode) --
                seeking = False
                _last_seek_time = 0.0

                def _fmt_time(seconds: float) -> str:
                    m, s = divmod(int(seconds), 60)
                    h, m = divmod(m, 60)
                    if h > 0:
                        return f"{h}:{m:02d}:{s:02d}"
                    return f"{m}:{s:02d}"

                def _on_seek_drag():
                    nonlocal seeking
                    seeking = True

                def _on_seek_release():
                    nonlocal seeking, _last_seek_time
                    seeking = False
                    _last_seek_time = 0.0
                    if _current_video_source is not None and pipeline is not None:
                        pipeline.seek_video(seek_slider.value)

                with ui.row().classes("w-full justify-center q-mt-xs") as seek_row:
                    seek_row.visible = False
                    seek_slider = (
                        ui.slider(
                            min=0,
                            max=1,
                            step=0.001,
                        )
                        .props("dense")
                        .classes("w-full")
                    )
                    seek_label = (
                        ui.label("0:00 / 0:00")
                        .classes("text-caption text-white q-ml-sm")
                        .style("min-width: 100px; text-align: right;")
                    )
                    seek_slider.on("update:model-value", _on_seek_drag)
                    seek_slider.on("update:model-value-end", _on_seek_release)

                with ui.row().classes("w-full justify-center q-mt-xs"):
                    with ui.element("div").classes("camera-toolbar"):
                        camera_toggle_button = ui.button(
                            "", on_click=on_toggle_camera
                        ).props("flat round dense size=sm color=negative icon=stop")
                        camera_toggle_button.tooltip("Start/Stop Camera")
                        open_video_btn = ui.button(
                            "", icon="movie", on_click=_pick_and_play
                        ).props("flat round dense size=sm color=white")
                        open_video_btn.tooltip("Open Video File")
                        youtube_btn = ui.button(
                            "", icon="smart_display", on_click=_pick_youtube
                        ).props("flat round dense size=sm color=white")
                        youtube_btn.tooltip("YouTube Stream")
                        youtube_loading = (
                            ui.spinner(size="sm")
                            .props("color=white")
                            .classes("q-ml-xs")
                        )
                        youtube_loading.visible = False
                        webcam_btn = ui.button(
                            "", icon="videocam", on_click=lambda: _switch_to_webcam()
                        ).props("flat round dense size=sm color=white")
                        webcam_btn.tooltip("Switch to Webcam")
                        webcam_btn.visible = False
                        video_label = (
                            ui.label("")
                            .classes("text-caption text-white q-ml-xs")
                            .style(
                                "max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
                            )
                        )
                        video_label.visible = False
                        roi_zoom_btn = ui.button(
                            "", icon="crop_free", on_click=on_roi_zoom
                        ).props("flat round dense size=sm color=white")
                        roi_zoom_btn.tooltip("Select Zoom Area")
                        ui.button(
                            "", icon="zoom_out_map", on_click=on_reset_zoom
                        ).props("flat round dense size=sm color=white").tooltip(
                            "Reset Zoom"
                        )

                        # -- Font button --
                        font_btn = (
                            ui.button(icon="text_fields")
                            .props("flat round dense size=sm color=white")
                            .style("min-width: 36px; min-height: 36px;")
                        )
                        font_btn.tooltip("Font scale")
                        with font_btn:
                            with ui.menu():
                                with ui.row().classes(
                                    "items-center q-pa-xs gap-2 flex-nowrap"
                                ):
                                    ui.slider(
                                        min=1.0,
                                        max=3.0,
                                        step=0.1,
                                        value=state.font_scale,
                                    ).bind_value_to(state, "font_scale").props(
                                        "dense"
                                    ).classes("w-24")
                                    ui.label().bind_text_from(
                                        state,
                                        "font_scale",
                                        backward=lambda v: f"{v:.1f}",
                                    ).classes("text-bold text-caption")

                        # -- Thickness button --
                        thick_btn = (
                            ui.button(icon="line_weight")
                            .props("flat round dense size=sm color=white")
                            .style("min-width: 36px; min-height: 36px;")
                        )
                        thick_btn.tooltip("Line thickness")
                        with thick_btn:
                            with ui.menu():
                                with ui.row().classes(
                                    "items-center q-pa-xs gap-2 flex-nowrap"
                                ):
                                    ui.slider(
                                        min=1,
                                        max=10,
                                        step=1,
                                        value=state.line_thickness,
                                    ).bind_value_to(state, "line_thickness").props(
                                        "dense"
                                    ).classes("w-24")
                                    ui.label().bind_text_from(
                                        state,
                                        "line_thickness",
                                        backward=lambda v: str(v),
                                    ).classes("text-bold text-caption")

                        # -- Color button (current color circle, click for palette) --
                        color_btn = (
                            ui.button("")
                            .props("round dense size=sm")
                            .style(
                                f"background: {color_name_to_hex(state.overlay_color_name)} !important; min-width: 36px; min-height: 36px;"
                            )
                        )
                        color_btn.tooltip("Overlay color")
                        with color_btn:
                            with ui.menu():
                                with ui.row().classes("q-pa-xs gap-1 flex-nowrap"):
                                    for color_name in COLOR_MAP:
                                        hex_color = color_name_to_hex(color_name)
                                        ui.button("").props(
                                            "round dense size=sm"
                                        ).style(
                                            f"background: {hex_color} !important; min-width: 32px; min-height: 32px;"
                                        ).on_click(
                                            lambda n=color_name: (
                                                setattr(state, "overlay_color_name", n),
                                                color_btn.style(
                                                    f"background: {color_name_to_hex(n)} !important; min-width: 36px; min-height: 36px;"
                                                ),
                                            )
                                        ).tooltip(color_name)

                        update_camera_toggle_button()

                # -- Identity panels (tracked faces + tracked bodies) --
                state.load_face_id_names()
                identity_panels_row = ui.row().classes("w-full q-mt-sm gap-2")
                identity_panels_row.visible = False

                with identity_panels_row:
                    with ui.element("div").classes("identity-panel"):
                        ui.label("Tracked faces").classes("identity-panel-title")
                        face_id_chip_row = ui.row().classes("w-full gap-1 flex-wrap")
                        face_empty = ui.label("No faces tracked").classes(
                            "identity-empty-state"
                        )
                    with ui.element("div").classes("identity-panel"):
                        ui.label("Tracked bodies").classes("identity-panel-title")
                        body_id_chip_row = ui.row().classes("w-full gap-1 flex-wrap")
                        body_empty = ui.label("No bodies tracked").classes(
                            "identity-empty-state"
                        )

                # -- Face naming dialog --
                add_name_dialog = ui.dialog()
                with add_name_dialog, ui.card().classes("w-80"):
                    ui.label("Save Face Name").classes("text-bold text-h6")
                    add_name_input = ui.input(
                        label="Name", placeholder="Enter a name..."
                    ).classes("w-full")
                    with ui.row().classes("w-full justify-end"):
                        ui.button("Cancel", on_click=add_name_dialog.close).props(
                            "flat"
                        )
                        save_name_button = ui.button("Save").props("color=primary")

                pending_face_track_id: int | None = None

                def save_face_name():
                    nonlocal current_face_chip_key, pending_face_track_id
                    name = (add_name_input.value or "").strip()
                    normalized_name = name.upper()
                    track_id = pending_face_track_id
                    if track_id is None:
                        return
                    if not name:
                        ui.notify("Enter a name", type="warning")
                        return
                    if pipeline is None:
                        ui.notify("Camera pipeline not ready", type="warning")
                        return
                    frame = pipeline.get_latest_frame_copy()
                    detection = pipeline.get_latest_face_detection(track_id)
                    if frame is None or detection is None:
                        ui.notify("Selected face is no longer active", type="warning")
                        return
                    pipeline.model.face_engine.enroll_identity(
                        normalized_name,
                        frame,
                        detection,
                        track_id,
                    )
                    # Save face thumbnail
                    x1, y1, x2, y2 = map(int, detection["bbox"])
                    thumb = frame[y1:y2, x1:x2]
                    if thumb.size > 0:
                        scale = min(
                            config.IDENTITY_THUMBNAIL_SIZE / thumb.shape[1],
                            config.IDENTITY_THUMBNAIL_SIZE / thumb.shape[0],
                        )
                        new_w = max(1, int(thumb.shape[1] * scale))
                        new_h = max(1, int(thumb.shape[0] * scale))
                        thumb_resized = cv2.resize(thumb, (new_w, new_h))
                        THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
                        safe_name = "".join(
                            c if c.isalnum() or c in "-_" else "_"
                            for c in normalized_name
                        )
                        cv2.imwrite(
                            str(THUMBNAILS_DIR / f"{safe_name}.jpg"), thumb_resized
                        )
                    state.set_face_id_name(track_id, normalized_name)
                    add_name_input.value = ""
                    pending_face_track_id = None
                    current_face_chip_key = ()
                    add_name_dialog.close()

                save_name_button.on_click(save_face_name)

                def open_add_name(track_id: int):
                    nonlocal pending_face_track_id
                    pending_face_track_id = track_id
                    add_name_input.value = state.face_id_names.get(track_id, "")
                    add_name_dialog.open()

                def delete_face_name(track_id: int):
                    nonlocal current_face_chip_key
                    if pipeline is None:
                        ui.notify("Camera pipeline not ready", type="warning")
                        return
                    name = state.face_id_names.get(track_id, "")
                    if not name:
                        return
                    pipeline.model.face_engine.remove_identity(name, track_id)
                    state.clear_face_id_name(track_id)
                    # Remove thumbnail
                    safe_name = "".join(
                        c if c.isalnum() or c in "-_" else "_" for c in name
                    )
                    thumb_path = THUMBNAILS_DIR / f"{safe_name}.jpg"
                    if thumb_path.exists():
                        thumb_path.unlink()
                    current_face_chip_key = ()

                # -- Body naming dialog --
                body_add_name_dialog = ui.dialog()
                with body_add_name_dialog, ui.card().classes("w-80"):
                    ui.label("Save Body Name").classes("text-bold text-h6")
                    body_add_name_input = ui.input(
                        label="Name", placeholder="Enter a name..."
                    ).classes("w-full")
                    with ui.row().classes("w-full justify-end"):
                        ui.button("Cancel", on_click=body_add_name_dialog.close).props(
                            "flat"
                        )
                        body_save_name_button = ui.button("Save").props("color=primary")

                pending_body_track_id: int | None = None

                def save_body_name():
                    nonlocal current_body_chip_key, pending_body_track_id
                    name = (body_add_name_input.value or "").strip()
                    normalized_name = name.upper()
                    track_id = pending_body_track_id
                    if track_id is None:
                        return
                    if not name:
                        ui.notify("Enter a name", type="warning")
                        return
                    if pipeline is None:
                        ui.notify("Camera pipeline not ready", type="warning")
                        return
                    snapshot = pipeline.get_latest_body_snapshot(track_id)
                    if snapshot is None:
                        ui.notify("Selected body is no longer active", type="warning")
                        return
                    frame = snapshot["frame"]
                    detection = snapshot["detection"]
                    pipeline.model.body_id_engine.enroll_identity(
                        normalized_name,
                        frame,
                        detection,
                        track_id,
                    )
                    # Save body thumbnail
                    x1, y1, x2, y2 = map(int, detection["bbox"])
                    thumb = frame[y1:y2, x1:x2]
                    if thumb.size > 0:
                        scale = min(
                            config.IDENTITY_THUMBNAIL_SIZE / thumb.shape[1],
                            config.IDENTITY_THUMBNAIL_SIZE / thumb.shape[0],
                        )
                        new_w = max(1, int(thumb.shape[1] * scale))
                        new_h = max(1, int(thumb.shape[0] * scale))
                        thumb_resized = cv2.resize(thumb, (new_w, new_h))
                        BODY_THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
                        safe_name = "".join(
                            c if c.isalnum() or c in "-_" else "_"
                            for c in normalized_name
                        )
                        cv2.imwrite(
                            str(BODY_THUMBNAILS_DIR / f"{safe_name}.jpg"),
                            thumb_resized,
                        )
                    state.set_body_id_name(track_id, normalized_name)
                    body_add_name_input.value = ""
                    pending_body_track_id = None
                    current_body_chip_key = ()
                    body_add_name_dialog.close()

                body_save_name_button.on_click(save_body_name)

                def open_body_add_name(track_id: int):
                    nonlocal pending_body_track_id
                    pending_body_track_id = track_id
                    body_add_name_input.value = state.body_id_names.get(track_id, "")
                    body_add_name_dialog.open()

                def delete_body_name(track_id: int):
                    nonlocal current_body_chip_key
                    if pipeline is None:
                        ui.notify("Camera pipeline not ready", type="warning")
                        return
                    name = state.body_id_names.get(track_id, "")
                    if not name:
                        return
                    pipeline.model.body_id_engine.remove_identity(name, track_id)
                    state.clear_body_id_name(track_id)
                    # Remove thumbnail
                    safe_name = "".join(
                        c if c.isalnum() or c in "-_" else "_" for c in name
                    )
                    thumb_path = BODY_THUMBNAILS_DIR / f"{safe_name}.jpg"
                    if thumb_path.exists():
                        thumb_path.unlink()
                    current_body_chip_key = ()

                # -- Shared chip rendering helper --
                def _identity_chip(
                    *,
                    track_id: int,
                    label: str,
                    tooltip_text: str,
                    thumb_url: str | None,
                    on_add: Callable[[int], None] | None,
                    on_delete: Callable[[int], None] | None,
                    border_color: str | None = None,
                ):
                    border_style = (
                        f"border: 3px solid {border_color};"
                        if border_color
                        else "border: 1px solid rgba(255,255,255,0.15);"
                    )
                    with (
                        ui.element("div")
                        .classes(
                            "identity-chip-container column no-wrap bg-grey-9 text-white"
                        )
                        .style(
                            f"gap: 0; {border_style} width: {config.IDENTITY_CHIP_WIDTH}px; height: {config.IDENTITY_CHIP_HEIGHT}px; flex-shrink: 0; overflow: hidden; border-radius: 8px;"
                        )
                    ):
                        ui.tooltip(tooltip_text)
                        if thumb_url:
                            ui.element("img").classes(
                                "identity-chip-img flex-none"
                            ).props(f'src="{thumb_url}"')
                        else:
                            ui.element("div").classes("identity-chip-placeholder")
                        with ui.element("div").classes(
                            "row items-center no-wrap q-px-xs flex-1 min-w-0 relative"
                        ):
                            ui.label(label).classes(
                                "text-caption ellipsis w-full min-w-0"
                            )
                            if on_delete:
                                ui.button(
                                    "",
                                    icon="close",
                                    on_click=lambda tid=track_id: on_delete(tid),
                                ).props("dense flat round size=sm color=red").classes(
                                    "identity-chip-action-btn flex-none"
                                ).style(
                                    "position: absolute; right: 0; top: 50%; transform: translateY(-50%);"
                                )
                            elif on_add:
                                ui.button(
                                    "",
                                    icon="add",
                                    on_click=lambda tid=track_id: on_add(tid),
                                ).props("dense flat round size=sm color=green").classes(
                                    "identity-chip-action-btn flex-none"
                                ).style(
                                    "position: absolute; right: 0; top: 50%; transform: translateY(-50%);"
                                )

                def rebuild_face_id_chips():
                    face_id_chip_row.clear()
                    active_ids = sorted(state.active_face_ids)
                    face_empty.visible = len(active_ids) == 0
                    face_body_links = state.get_face_body_links_snapshot()
                    with face_id_chip_row:
                        for track_id in active_ids:
                            face_name = state.face_id_names.get(track_id, "")
                            body_tid = face_body_links.get(track_id)
                            if face_name:
                                label = face_name
                            elif body_tid is not None and state.body_id_names.get(
                                body_tid
                            ):
                                label = state.body_id_names[body_tid]
                            else:
                                label = f"ID: {track_id}"
                            thumb_url = (
                                _face_thumbnail_url(face_name) if face_name else None
                            )
                            border_color = None
                            if body_tid is not None:
                                shared_name = _shared_identity_for_link(
                                    track_id, body_tid
                                )
                                border_color = _chip_link_color(body_tid, shared_name)
                            _identity_chip(
                                track_id=track_id,
                                label=label,
                                tooltip_text=f"track_id: {track_id}",
                                thumb_url=thumb_url,
                                on_add=None if face_name else open_add_name,
                                on_delete=delete_face_name if face_name else None,
                                border_color=border_color,
                            )

                def rebuild_body_id_chips():
                    body_id_chip_row.clear()
                    active_ids = sorted(state.active_body_ids)
                    body_empty.visible = len(active_ids) == 0
                    face_body_links = state.get_face_body_links_snapshot()
                    body_face_links = {b: f for f, b in face_body_links.items()}
                    with body_id_chip_row:
                        for track_id in active_ids:
                            body_name = state.body_id_names.get(track_id, "")
                            face_tid = body_face_links.get(track_id)
                            if body_name:
                                label = body_name
                            elif face_tid is not None and state.face_id_names.get(
                                face_tid
                            ):
                                label = state.face_id_names[face_tid]
                            else:
                                label = f"ID: {track_id}"
                            thumb_url = (
                                _body_thumbnail_url(body_name) if body_name else None
                            )
                            border_color = None
                            if face_tid is not None:
                                shared_name = _shared_identity_for_link(
                                    face_tid, track_id
                                )
                                border_color = _chip_link_color(track_id, shared_name)
                            _identity_chip(
                                track_id=track_id,
                                label=label,
                                tooltip_text=f"body_track_id: {track_id}",
                                thumb_url=thumb_url,
                                on_add=None if body_name else open_body_add_name,
                                on_delete=delete_body_name if body_name else None,
                                border_color=border_color,
                            )

            # -- controls (right on desktop, bottom on mobile) --
            with ui.column().style("flex: 1 1 0; min-width: 0; overflow: hidden;"):
                ui.element("div").style("height: 56px")  # spacer to align with title

                # Search -------------------------------------------------------
                def on_tab(tab):
                    if tab == "Detect":
                        state.set_mode("everything")
                        state.submit_target("")
                        set_search_status("")
                    elif tab == "Face":
                        state.set_mode("face")
                        state.submit_target("")
                        set_search_status("Face Mesh Active")
                    else:
                        state.set_mode("find")

                with ui.card().classes("w-full q-pa-none control-card"):
                    with ui.tabs().classes("w-full mode-tabs") as tabs:
                        ui.tab("Find", icon="search")
                        ui.tab("Detect", icon="visibility")
                        ui.tab("Face", icon="face")
                    tabs.on_value_change(lambda e: on_tab(e.value))

                    with ui.tab_panels(tabs, value="Find").classes(
                        "w-full clean-panels"
                    ):
                        with ui.tab_panel("Find"):
                            with ui.row().classes(IWN):
                                search_inp = (
                                    ui.input(placeholder='e.g. "Where is my red mug?"')
                                    .classes(GROW)
                                    .on("keydown.enter", lambda: do_search())
                                )
                                ui.button("Find", on_click=lambda: do_search()).props(
                                    "unelevated no-caps"
                                ).classes("premium-button")
                            # Quick category pills (flex-wrap, mutex)
                            CATEGORIES = [
                                "person",
                                "animal",
                                "plant",
                                "food",
                                "fruit",
                                "beverage",
                                "furniture",
                                "chair",
                                "electronics",
                                "phone",
                                "laptop",
                                "clothing",
                                "shoe",
                                "bag",
                                "book",
                                "toy",
                                "bottle",
                                "cup",
                                "vehicle",
                                "lamp",
                            ]
                            selected_cat: str | None = None
                            cat_btns: dict = {}
                            with ui.row().classes("w-full q-mt-xs gap-1 flex-wrap"):
                                for cat in CATEGORIES:
                                    btn = ui.button(
                                        cat,
                                        on_click=lambda c=cat: toggle_cat(c),
                                    ).props("outline rounded no-caps")
                                    cat_btns[cat] = btn

                            def toggle_cat(cat: str):
                                nonlocal selected_cat
                                if cat == selected_cat:
                                    return  # already active
                                # Deselect previous
                                if selected_cat is not None:
                                    cat_btns[selected_cat].props("outline")
                                # Select new
                                selected_cat = cat
                                cat_btns[cat].props(remove="outline")
                                target = normalize_query(cat)
                                state.submit_target(target)
                                set_search_status(f'Searching: "{cat}"')
                                search_inp.value = ""
                                if pipeline is not None:
                                    pipeline.model.set_prompt(target)

                            ui.element("div").classes("flex-grow")  # spacer
                            # Search status chip (always at bottom)
                            with ui.row().classes(
                                "items-center q-mt-xs gap-1"
                            ) as search_status_row:
                                ui.icon("search").classes("text-primary text-caption")
                                search_status = ui.label("").classes(
                                    "text-caption text-white"
                                )
                            search_status_row.visible = False
                        with ui.tab_panel("Detect"):
                            with ui.row().classes(IWN):
                                ui.label("Top").classes(CAP)
                                top_slider = ui.slider(
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=5,
                                    on_change=lambda e: top_label.set_text(
                                        f"{int(e.value or 5)}"
                                    ),
                                ).classes(GROW)
                                top_slider.bind_value_to(state, "top_labels")
                                top_label = ui.label("5").classes("text-bold q-ml-sm")
                                ui.label("items").classes(CAP)

                            with ui.row().classes(IWN):
                                ui.label("Threshold").classes(CAP)
                                ui.slider(
                                    min=config.CONFIDENCE_MIN,
                                    max=config.CONFIDENCE_MAX,
                                    step=config.CONFIDENCE_STEP,
                                    value=config.DEFAULT_THRESHOLD,
                                    on_change=lambda e: (
                                        state.set_confidence(e.value),
                                        thresh_label.set_text(
                                            f"{(e.value or 0.0) * 100:.0f}%"
                                        ),
                                    ),
                                ).classes(GROW)
                                thresh_label = ui.label(
                                    f"{config.DEFAULT_THRESHOLD * 100:.0f}%"
                                ).classes("text-bold q-ml-sm")

                        with ui.tab_panel("Face"):
                            ui.label("Privacy mode").classes(CAP)
                            with ui.row().classes(IWN):
                                ui.toggle(
                                    ["None", "Pixelate", "Gaussian"],
                                    value="None",
                                    on_change=lambda e: setattr(
                                        state, "privacy_mode", e.value
                                    ),
                                ).props("dense spread").classes("w-full")
                            ui.label("Image filter").classes(CAP)
                            with ui.row().classes(IWN):
                                ui.toggle(
                                    ["None", "Heat", "CRT", "Comic", "Invert"],
                                    value="None",
                                    on_change=lambda e: setattr(
                                        state, "visual_filter", e.value
                                    ),
                                ).props("dense spread").classes("w-full")
                            with ui.row().classes(IWN):
                                ui.switch("Tracking").bind_value_to(
                                    state, "tracking_enabled"
                                )
                            with ui.row().classes(IWN):
                                ui.switch("Face mesh").bind_value_to(
                                    state, "face_mesh_enabled"
                                )
                            with ui.row().classes(IWN):
                                ui.switch("Body mesh").bind_value_to(
                                    state, "body_mesh_enabled"
                                )

    # ---- Timer-driven refresh ------------------------------------------------
    _clear_display()

    def refresh_all():
        nonlocal \
            current_face_chip_key, \
            current_body_chip_key, \
            current_frame_jpeg, \
            _last_seek_time
        update_camera_toggle_button()
        if pipeline is not None:
            jpeg = pipeline.get_latest_encoded_frame()
            if jpeg is not None and jpeg != current_frame_jpeg:
                current_frame_jpeg = jpeg
                encoded = base64.b64encode(jpeg).decode("ascii")
                cam.set_source(f"data:image/jpeg;base64,{encoded}")
        show_panels = state.mode == "face" and state.tracking_enabled
        if show_panels != identity_panels_row.visible:
            identity_panels_row.visible = show_panels
        # Video seek bar progress
        if isinstance(_current_video_source, VideoFilePlayer):
            if seeking:
                now = time.time()
                if now - _last_seek_time > 0.25:
                    _last_seek_time = now
                    if pipeline is not None:
                        pipeline.seek_video(seek_slider.value)
            else:
                seek_slider.value = _current_video_source.progress
            seek_label.text = (
                f"{_fmt_time(_current_video_source.current_time)}"
                f" / {_fmt_time(_current_video_source.duration)}"
            )
        else:
            if seek_row.visible:
                seek_row.visible = False
        if show_panels:
            links_dict = state.get_face_body_links_snapshot()
            links_snapshot = tuple(sorted(links_dict.items()))
            face_chip_key = (
                tuple(
                    (
                        track_id,
                        state.face_id_names.get(track_id, ""),
                        state.body_id_names.get(links_dict.get(track_id), ""),
                    )
                    for track_id in sorted(state.active_face_ids)
                ),
                links_snapshot,
            )
            if face_chip_key != current_face_chip_key:
                current_face_chip_key = face_chip_key
                rebuild_face_id_chips()
            body_to_face = {b: f for f, b in links_dict.items()}
            body_chip_key = (
                tuple(
                    (
                        track_id,
                        state.body_id_names.get(track_id, ""),
                        state.face_id_names.get(body_to_face.get(track_id), ""),
                    )
                    for track_id in sorted(state.active_body_ids)
                ),
                links_snapshot,
            )
            if body_chip_key != current_body_chip_key:
                current_body_chip_key = body_chip_key
                rebuild_body_id_chips()
        update_search_status()

    ui.timer(1 / 30, refresh_all)

    # ---- Page-local helper closures -----------------------------------------

    def do_search():
        nonlocal selected_cat
        raw = (search_inp.value or "").strip()
        target = normalize_query(raw)
        if target:
            # Clear category selection when user types
            if selected_cat is not None:
                cat_btns[selected_cat].props("outline")
                selected_cat = None
            state.submit_target(target)
            set_search_status(f'Searching: "{target}"')
            search_inp.value = ""
            if pipeline is not None:
                pipeline.model.set_prompt(target)
        else:
            ui.notify("Enter what you want to find", type="warning")

    def update_search_status():
        if state.mode == "face":
            set_search_status("Face Mesh Active")
        elif state.submitted_target:
            set_search_status(f'Searching: "{state.submitted_target}"')
        else:
            set_search_status("")


# ---- Lifecycle --------------------------------------------------------------


def start_services():
    """Load models on startup. Pipeline is started by the user via the Play button."""
    global pipeline, device_str, bundle

    device_str = get_device()
    logger.info("Starting services (device=%s)", device_str)

    try:
        bundle = load_model_bundle(config.PROMPTED_MODEL, config.PROMPTFREE_MODEL)
        state.set_models_ready(True)
        state.set_models_error(None)
        logger.info("Models loaded successfully")
    except Exception as exc:
        bundle = None
        msg = f"Model loading failed: {exc}"
        logger.error(msg)
        state.set_models_error(msg)


def stop_services():
    """Release camera and join worker threads."""
    logger.info("Shutting down services")
    if pipeline is not None:
        pipeline.stop()


def on_connect():
    global _session_count
    _session_count += 1


def on_disconnect():
    global _session_count, pipeline
    _session_count = max(0, _session_count - 1)
    if _session_count == 0 and pipeline is not None:
        logger.info("Last client disconnected, stopping pipeline")
        pipeline.stop()
        pipeline = None


napp.on_startup(lambda: threading.Thread(target=start_services, daemon=True).start())
napp.on_shutdown(stop_services)
napp.on_connect(on_connect)
napp.on_disconnect(on_disconnect)


# ---- Entry point ------------------------------------------------------------


def main():
    """Launch the NiceGUI server on localhost."""
    ui.run(
        title=config.TITLE,
        host=config.HOST,
        port=config.PORT,
        show=False,
        reload=True,
    )
