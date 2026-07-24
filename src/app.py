"""NiceGUI web interface and application lifecycle."""

from __future__ import annotations

import logging
import threading
import time
import warnings

warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from fastapi.responses import Response
from nicegui import app as napp
from nicegui import ui

from src import config
from src.models import get_device, load_model_bundle
from src.pipeline import CapturePipeline
from src.state import AppState
from src.utils import normalize_query

logger = logging.getLogger(__name__)

# Module-level references shared between UI and lifecycle hooks.
state: AppState = AppState()
pipeline: CapturePipeline | None = None
device_str: str = "unknown"


@napp.get("/frame.jpg")
async def serve_frame():
    """Return the latest annotated JPEG frame."""
    if not state.models_ready:
        return Response(status_code=503)
    if pipeline is None:
        return Response(status_code=503)
    jpeg = pipeline.get_latest_encoded_frame()
    if jpeg is None:
        return Response(status_code=204)
    return Response(content=jpeg, media_type="image/jpeg")


@ui.page("/")
def index():
    """Assemble the full page."""
    ui.dark_mode().enable()

    FW = "w-full"
    IWN = "items-center w-full no-wrap"
    CAP = "text-caption"
    GROW = "flex-grow"

    # ---- page chrome --------------------------------------------------------
    ui.add_head_html("""
    <style>
      .q-tab .q-tab__content { flex-direction: row; gap: 6px; }
      .q-tab .q-tab__icon { margin: 0; }
    </style>
    """)

    with ui.element("div").classes(FW).style("max-width: 95%; margin: 24px auto;"):
        with ui.row().classes("w-full no-wrap"):
            # -- left column: controls --
            with ui.column().classes("col-12 col-md-3"):
                ui.element("div").style("height: 56px")  # spacer to align with title

                # Search -------------------------------------------------------
                def on_tab(tab):
                    if tab == "Detect":
                        state.set_mode("everything")
                        state.submit_target("")
                        search_status.set_text("")
                    elif tab == "Face":
                        state.set_mode("face")
                        state.submit_target("")
                        search_status.set_text("Face Mesh Active")
                    else:
                        state.set_mode("find")

                with ui.card().classes("w-full q-pa-none"):
                    with ui.tabs().classes("w-full") as tabs:
                        ui.tab("Find", icon="search").classes("w-full")
                        ui.tab("Detect", icon="visibility").classes("w-full")
                        ui.tab("Face", icon="face").classes("w-full")
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
                            with ui.row().classes("w-full no-wrap"):
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
                                ui.switch("Text labels").bind_value_to(
                                    state, "face_show_labels"
                                )

            # -- right column: live camera feed --
            with ui.column().classes("col-12 col-md-9"):
                ui.label("Real-time object detection").classes(
                    "text-h4 text-weight-bold text-center w-full q-mb-md"
                )
                cam = ui.interactive_image().classes("w-full border-1 rounded")

    # ---- Timer-driven refresh ------------------------------------------------
    def refresh_all():
        cam.set_source(f"/frame.jpg?{int(time.time() * 1000)}")
        update_search_status()

    ui.timer(0.001, refresh_all)

    # ---- Page-local helper closures -----------------------------------------

    def do_search():
        raw = (search_inp.value or "").strip()
        target = normalize_query(raw)
        if target:
            state.submit_target(target)
            search_status.set_text(f'Searching: "{target}"')
            search_inp.value = ""
            if pipeline is not None:
                pipeline.model.set_prompt(target)
        else:
            ui.notify("Enter what you want to find", type="warning")

    def update_search_status():
        if state.mode == "face":
            search_status.set_text("Face Mesh Active")
        elif state.submitted_target:
            search_status.set_text(f'Searching: "{state.submitted_target}"')
        else:
            search_status.set_text("")


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
