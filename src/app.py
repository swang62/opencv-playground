"""NiceGUI web interface and application lifecycle."""

from __future__ import annotations

import base64
import logging
import threading
import warnings

warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from nicegui import app as napp
from nicegui import ui

from src import config
from src.models import get_device, load_model_bundle
from src.pipeline import CapturePipeline
from src.state import COLOR_MAP, AppState, color_name_to_hex
from src.utils import normalize_query

logger = logging.getLogger(__name__)

# Module-level references shared between UI and lifecycle hooks.
state: AppState = AppState()
pipeline: CapturePipeline | None = None
device_str: str = "unknown"


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

    def set_search_status(text: str):
        nonlocal current_search_status
        if text == current_search_status:
            return
        current_search_status = text
        search_status.set_text(text)

    # ---- page chrome --------------------------------------------------------
    ui.add_head_html("""
    <style>
      .q-tab { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex-shrink: 1; min-width: 0; }
      .q-tab .q-tab__content { flex-direction: row; gap: 4px; }
      .q-tab .q-tab__icon { margin: 0; }
    </style>
    """)

    with ui.element("div").classes(FW).style("max-width: 98%; margin: 24px auto;"):
        with ui.row().classes("w-full flex-wrap items-start"):
            # -- webcam (left on desktop, top on mobile) --
            with ui.column().classes("flex-1 min-w-0"):
                ui.label(config.TITLE).classes(
                    "text-h4 text-weight-bold text-center w-full q-mb-md"
                )
                cam = ui.interactive_image().classes("w-full border-1 rounded")

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

                with ui.card().classes("w-full q-pa-none"):
                    with ui.tabs().classes("w-full") as tabs:
                        ui.tab("Find", icon="search")
                        ui.tab("Detect", icon="visibility")
                        ui.tab("Face", icon="face")
                    tabs.on_value_change(lambda e: on_tab(e.value))

                    with ui.tab_panels(tabs, value="Find").classes("w-full"):
                        with ui.tab_panel("Find"):
                            with ui.row().classes(IWN):
                                search_inp = (
                                    ui.input(placeholder='e.g. "Where is my red mug?"')
                                    .classes(GROW)
                                    .on("keydown.enter", lambda: do_search())
                                )
                                ui.button("Find", on_click=lambda: do_search()).props(
                                    "flat color=primary dense"
                                )
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
                                    value=config.DEFAULT_CONFIDENCE,
                                    on_change=lambda e: (
                                        state.set_confidence(e.value),
                                        thresh_label.set_text(
                                            f"{(e.value or 0.0) * 100:.0f}%"
                                        ),
                                    ),
                                ).classes(GROW)
                                thresh_label = ui.label(
                                    f"{config.DEFAULT_CONFIDENCE * 100:.0f}%"
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
                                    label="Video filter",
                                    on_change=lambda e: setattr(
                                        state, "visual_filter", e.value
                                    ),
                                ).classes("w-full")
                            with ui.row().classes(IWN):
                                ui.switch("Face mesh").bind_value_to(
                                    state, "face_show_wireframe"
                                )
                            with ui.row().classes(IWN):
                                ui.switch("Body mesh").bind_value_to(
                                    state, "face_show_skeleton"
                                )
                            with ui.row().classes(IWN):
                                ui.switch("Head direction").bind_value_to(
                                    state, "face_show_headpose"
                                )
                            with ui.row().classes(IWN):
                                ui.switch("Labels").bind_value_to(
                                    state, "face_show_labels"
                                )

                # -- Global settings below tabs --
                with ui.card().classes("w-full q-pa-md"):
                    ui.label("Global").classes("text-bold text-h6")
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
        nonlocal current_frame_jpeg
        if pipeline is not None:
            jpeg = pipeline.get_latest_encoded_frame()
            if jpeg is not None and jpeg != current_frame_jpeg:
                current_frame_jpeg = jpeg
                encoded = base64.b64encode(jpeg).decode("ascii")
                cam.set_source(f"data:image/jpeg;base64,{encoded}")
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
    global pipeline, device_str

    device_str = get_device()
    logger.info("Starting services (device=%s)", device_str)

    try:
        bundle = load_model_bundle(config.PROMPTED_MODEL, config.PROMPTFREE_MODEL)
        state.set_models_ready(True)
        state.set_models_error(None)
        logger.info("Models loaded successfully")
    except Exception as exc:
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
