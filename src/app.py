"""NiceGUI web interface and application lifecycle."""

from __future__ import annotations

import base64
import logging
import threading
import warnings
from pathlib import Path

import cv2

warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from nicegui import app as napp
from nicegui import ui

from src import config
from src.models import ModelBundle, get_device, load_model_bundle
from src.pipeline import CapturePipeline
from src.state import COLOR_MAP, AppState, color_name_to_hex
from src.utils import normalize_query

logger = logging.getLogger(__name__)

# Module-level references shared between UI and lifecycle hooks.
state: AppState = AppState()
pipeline: CapturePipeline | None = None
bundle: ModelBundle | None = None
device_str: str = "unknown"
THUMBNAILS_DIR = Path(config.MODELS_DIR) / "screenshots"


def _face_thumbnail_url(name: str) -> str | None:
    """Return a data URL for the face thumbnail, or None if missing."""
    path = THUMBNAILS_DIR / f"{name}.jpg"
    if not path.exists():
        return None
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


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
    current_face_chip_key: tuple[tuple[int, str], ...] = ()

    def set_search_status(text: str):
        nonlocal current_search_status
        if text == current_search_status:
            return
        current_search_status = text
        search_status.set_text(text)

    # Core webcam lifecycle (defined early so all controls can reference them)
    def _start():
        global pipeline, bundle
        if bundle is None:
            ui.notify("Models not loaded", type="warning")
            return
        if pipeline is not None:
            return
        pipeline = CapturePipeline(bundle, state)
        pipeline.start()

    def _stop():
        global pipeline
        if pipeline is not None:
            pipeline.stop()
            pipeline = None

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
      .face-chip-img {{
        width: 100% !important;
        height: {config.FACE_CHIP_WIDTH}px !important;
        object-fit: cover;
        display: block;
        flex-shrink: 0;
      }}
      .face-chip-action-btn {{
        opacity: 0;
        transition: opacity 0.15s;
        background: rgba(10, 10, 14, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #fff !important;
      }}
      .face-chip-container:hover .face-chip-action-btn {{ opacity: 1; }}
      .face-chip-placeholder {{ width: 100%; height: {config.FACE_CHIP_WIDTH}px; background: #fff; flex-shrink: 0; }}
    </style>
    """)

    with (
        ui.element("div")
        .classes(FW)
        .style(
            f"max-width: {config.PAGE_MAX_WIDTH}px; margin: 0 auto; padding: {config.PAGE_PADDING_VERTICAL}px {config.PAGE_PADDING_HORIZONTAL}px;"
        )
    ):
        with ui.row().classes("w-full flex-wrap items-start"):
            # -- webcam (left on desktop, top on mobile) --
            with ui.column().classes("flex-1 min-w-0"):
                ui.label(config.TITLE).classes(
                    "text-h4 text-weight-bold text-center w-full q-mb-md"
                )
                cam = (
                    ui.interactive_image(
                        cross=f"{color_name_to_hex(state.overlay_color_name)}55",
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

                def update_crosshair_color():
                    cam.props(f"cross={color_name_to_hex(state.overlay_color_name)}55")

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
                    if pipeline is None:
                        _start()
                    else:
                        _stop()

                def update_camera_toggle_button():
                    if pipeline is None:
                        camera_toggle_button.props(
                            "flat round dense size=sm color=positive icon=play_arrow"
                        )
                    else:
                        camera_toggle_button.props(
                            "flat round dense size=sm color=negative icon=stop"
                        )

                with ui.row().classes("w-full justify-center q-mt-xs"):
                    with ui.element("div").classes("camera-toolbar"):
                        camera_toggle_button = ui.button(
                            "", on_click=on_toggle_camera
                        ).props("flat round dense size=sm color=negative icon=stop")
                        camera_toggle_button.tooltip("Start/Stop Camera")
                        roi_zoom_btn = ui.button(
                            "", icon="crop_free", on_click=on_roi_zoom
                        ).props("flat round dense size=sm color=white")
                        roi_zoom_btn.tooltip("Select Zoom Area")
                        ui.button(
                            "", icon="zoom_out_map", on_click=on_reset_zoom
                        ).props("flat round dense size=sm color=white").tooltip(
                            "Reset Zoom"
                        )
                        update_crosshair_color()
                        update_camera_toggle_button()

                # -- Face ID chips (under webcam) --
                state.load_face_id_names()
                face_id_chip_row = ui.row().classes("w-full q-mt-sm gap-2 flex-wrap")
                face_id_chip_row.visible = False

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
                            config.FACE_THUMBNAIL_SIZE / thumb.shape[1],
                            config.FACE_THUMBNAIL_SIZE / thumb.shape[0],
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

                def rebuild_face_id_chips():
                    face_id_chip_row.clear()
                    active_ids = sorted(state.active_face_ids)
                    with face_id_chip_row:
                        for track_id in active_ids:
                            face_name = state.face_id_names.get(track_id, "")
                            label = face_name or f"ID: {track_id}"
                            tooltip_text = f"track_id: {track_id}"
                            with (
                                ui.element("div")
                                .classes(
                                    "face-chip-container column no-wrap bg-grey-9 text-white"
                                )
                                .style(
                                    f"gap: 0; border: 1px solid rgba(255,255,255,0.15); width: {config.FACE_CHIP_WIDTH}px; height: {config.FACE_CHIP_HEIGHT}px; flex-shrink: 0; overflow: hidden; border-radius: 8px;"
                                )
                            ):
                                ui.tooltip(tooltip_text)
                                thumb_url = (
                                    _face_thumbnail_url(face_name)
                                    if face_name
                                    else None
                                )
                                if thumb_url:
                                    ui.element("img").classes(
                                        "face-chip-img flex-none"
                                    ).props(f'src="{thumb_url}"')
                                else:
                                    ui.element("div").classes("face-chip-placeholder")
                                with ui.element("div").classes(
                                    "row items-center no-wrap q-px-xs flex-1 min-w-0 relative"
                                ):
                                    ui.label(label).classes(
                                        "text-caption ellipsis w-full min-w-0"
                                    )
                                    if face_name:
                                        ui.button(
                                            "",
                                            icon="close",
                                            on_click=lambda _, tid=track_id: (
                                                delete_face_name(tid)
                                            ),
                                        ).props(
                                            "dense flat round size=sm color=red"
                                        ).classes(
                                            "face-chip-action-btn flex-none"
                                        ).style(
                                            "position: absolute; right: 0; top: 50%; transform: translateY(-50%);"
                                        )
                                    else:
                                        ui.button(
                                            "",
                                            icon="add",
                                            on_click=lambda _, tid=track_id: (
                                                open_add_name(tid)
                                            ),
                                        ).props(
                                            "dense flat round size=sm color=green"
                                        ).classes(
                                            "face-chip-action-btn flex-none"
                                        ).style(
                                            "position: absolute; right: 0; top: 50%; transform: translateY(-50%);"
                                        )

            # -- controls (right on desktop, bottom on mobile) --
            with ui.column().classes("flex-none"):
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

                def set_facial_recognition(enabled: bool):
                    state.face_show_labels = enabled
                    state.face_show_ids = enabled

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
                            search_status = ui.label("").classes(
                                "text-caption text-grey q-mt-n2"
                            )
                            with ui.row().classes(IWN):
                                ui.label("Opacity").classes(CAP)
                                opacity_slider = ui.slider(
                                    min=0.05,
                                    max=0.5,
                                    step=0.05,
                                    value=state.mask_opacity,
                                    on_change=lambda e: opacity_label.set_text(
                                        f"{(e.value or 0.0) * 100:.0f}%"
                                    ),
                                ).classes(GROW)
                                opacity_slider.bind_value_to(state, "mask_opacity")
                                opacity_label = ui.label(
                                    f"{state.mask_opacity * 100:.0f}%"
                                ).classes("text-bold q-ml-sm")

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
                            with ui.row().classes(IWN):
                                ui.select(
                                    [
                                        "None",
                                        "Sketch",
                                        "Thermal",
                                        "CRT",
                                        "Comic",
                                        "Invert",
                                    ],
                                    value="None",
                                    label="Image filter",
                                    on_change=lambda e: setattr(
                                        state, "visual_filter", e.value
                                    ),
                                ).classes("w-full")
                            with ui.row().classes(IWN):
                                ui.switch(
                                    "Tracking",
                                    value=state.face_show_labels
                                    and state.face_show_ids,
                                    on_change=lambda e: set_facial_recognition(
                                        bool(e.value)
                                    ),
                                )
                            with ui.row().classes(IWN):
                                ui.switch("Face mesh").bind_value_to(
                                    state, "face_show_wireframe"
                                )
                            with ui.row().classes(IWN):
                                ui.switch("Body mesh").bind_value_to(
                                    state, "face_show_skeleton"
                                )
                # -- Global settings below tabs --
                with ui.card().classes("w-full q-pa-md control-card"):
                    ui.label("Global settings").classes("text-bold text-h6")

                    # Webcam control strip
                    # with ui.row().classes("w-full items-center no-wrap gap-2"):
                    #     webcam_btn = (
                    #         ui.button("Refresh").props("outline").classes("flex-1")
                    #     )

                    #     def on_webcam_start_restart():
                    #         _start_restart()
                    #         webcam_btn.text = "Refresh"

                    #     webcam_btn.on_click(on_webcam_start_restart)

                    #     def on_webcam_stop():
                    #         _stop()
                    #         webcam_btn.text = "Start"

                    #     ui.button("Stop", on_click=on_webcam_stop).props(
                    #         "outline"
                    #     ).classes("flex-1")

                    with ui.row().classes(IWN):
                        ui.label("Font").classes(CAP)
                        font_slider = ui.slider(
                            min=1.0,
                            max=3.0,
                            step=0.1,
                            value=state.font_scale,
                            on_change=lambda e: font_label.set_text(f"{e.value:.1f}"),
                        ).classes(GROW)
                        font_slider.bind_value_to(state, "font_scale")
                        font_label = ui.label(f"{state.font_scale:.1f}").classes(
                            "text-bold q-ml-sm"
                        )
                    with ui.row().classes(IWN):
                        ui.label("Thickness").classes(CAP)
                        thick_slider = ui.slider(
                            min=1,
                            max=8,
                            step=1,
                            value=state.line_thickness,
                            on_change=lambda e: thick_label.set_text(
                                f"{int(e.value or 1)}"
                            ),
                        ).classes(GROW)
                        thick_slider.bind_value_to(state, "line_thickness")
                        thick_label = ui.label(f"{state.line_thickness}").classes(
                            "text-bold q-ml-sm"
                        )
                    with ui.row().classes("items-center w-full no-wrap justify-evenly"):
                        for name in COLOR_MAP:
                            bg = color_name_to_hex(name)
                            ui.button(
                                "",
                                on_click=lambda n=name: setattr(
                                    state, "overlay_color_name", n
                                ),
                            ).props("dense flat padding=none").style(
                                f"background: {bg}; width: 20px; height: 20px; min-width: 20px; border-radius: 4px;"
                            )

    # ---- Timer-driven refresh ------------------------------------------------
    def refresh_all():
        nonlocal current_face_chip_key, current_frame_jpeg
        update_crosshair_color()
        update_camera_toggle_button()
        if pipeline is not None:
            jpeg = pipeline.get_latest_encoded_frame()
            if jpeg is not None and jpeg != current_frame_jpeg:
                current_frame_jpeg = jpeg
                encoded = base64.b64encode(jpeg).decode("ascii")
                cam.set_source(f"data:image/jpeg;base64,{encoded}")
        show_chips = state.mode == "face" and state.face_show_ids
        if show_chips != face_id_chip_row.visible:
            face_id_chip_row.visible = show_chips
        if show_chips:
            face_chip_key = tuple(
                (track_id, state.face_id_names.get(track_id, ""))
                for track_id in sorted(state.active_face_ids)
            )
            if face_chip_key != current_face_chip_key:
                current_face_chip_key = face_chip_key
                rebuild_face_id_chips()
        update_search_status()

    ui.timer(1 / 30, refresh_all)

    # ---- Page-local helper closures -----------------------------------------

    def do_search():
        raw = (search_inp.value or "").strip()
        target = normalize_query(raw)
        if target:
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
    """Load models, create pipeline, start capture and inference."""
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
        return

    pipeline = CapturePipeline(bundle, state)
    pipeline.start()


def stop_services():
    """Release camera and join worker threads."""
    logger.info("Shutting down services")
    if pipeline is not None:
        pipeline.stop()


napp.on_startup(lambda: threading.Thread(target=start_services, daemon=True).start())
napp.on_shutdown(stop_services)


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
