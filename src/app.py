"""NiceGUI web interface and application lifecycle."""

from __future__ import annotations

import logging
import threading
import time

from fastapi.responses import Response
from nicegui import app as napp
from nicegui import ui

from src import config
from src.models import get_device, load_model_bundle
from src.pipeline import CapturePipeline
from src.state import AppState
from utils import normalize_query

logger = logging.getLogger(__name__)

# Module-level references shared between UI and lifecycle hooks.
_state: AppState = AppState()
_pipeline: CapturePipeline | None = None
_device_str: str = "unknown"


@napp.get("/frame.jpg")
async def serve_frame():
    """Return the latest annotated JPEG frame."""
    if not _state.models_ready:
        return Response(status_code=503)
    if _pipeline is None:
        return Response(status_code=503)
    jpeg = _pipeline.get_latest_jpeg()
    if jpeg is None:
        return Response(status_code=204)
    return Response(content=jpeg, media_type="image/jpeg")


@ui.page("/")
def index():
    """Assemble the full page."""
    ui.dark_mode().enable()

    _FW = "w-full"
    _IWN = "items-center w-full no-wrap"
    _H6 = "text-h6"
    _CAP = "text-caption"
    _GROW = "flex-grow"

    # ---- page chrome --------------------------------------------------------
    ui.add_head_html("""
    <style>
      .q-tab--active { border: 2px solid var(--q-primary); border-radius: 8px 8px 0 0; }
      .q-tab { border-bottom: 2px solid transparent; }
      .q-tab .q-tab__content { flex-direction: row; gap: 6px; }
      .q-tab .q-tab__icon { margin: 0; }
    </style>
    """)
    ui.label("Real-time Object Detection").classes(
        "text-h4 text-weight-bold q-mt-md q-mb-md"
    )

    with ui.element("div").classes(_FW).style("max-width: 95%; margin: 0 auto;"):
        with ui.row().classes("w-full no-wrap"):
            # -- left column: live camera feed --
            with ui.column().classes("col-12 col-md-9"):
                cam = ui.interactive_image().classes("w-full border-1 rounded")

            # -- right column: controls --
            with ui.column().classes("col-12 col-md-3"):
                # Search -------------------------------------------------------
                def _on_tab(tab):
                    if tab == "Detect":
                        _state.set_mode("everything")
                        _state.submit_target("")
                        _search_status.set_text("")
                    elif tab == "Face":
                        _state.set_mode("face")
                        _state.submit_target("")
                        _search_status.set_text("Face Mesh Active")
                    else:
                        _state.set_mode("find")

                with ui.card().classes("w-full q-pa-none"):
                    with ui.tabs().classes("w-full") as _tabs:
                        ui.tab("Find", icon="search").classes("w-full")
                        ui.tab("Detect", icon="visibility").classes("w-full")
                        ui.tab("Face", icon="face").classes("w-full")
                    _tabs.on_value_change(lambda e: _on_tab(e.value))

                    with ui.tab_panels(_tabs, value="Find").classes("w-full"):
                        with ui.tab_panel("Find"):
                            with ui.row().classes(_IWN):
                                search_inp = (
                                    ui.input(placeholder='e.g. "Where is my red mug?"')
                                    .classes(_GROW)
                                    .on("keydown.enter", lambda: _do_search())
                                )
                                _find_btn = ui.button(
                                    "Find", on_click=lambda: _do_search()
                                ).props("flat color=primary dense")
                            _search_status = ui.label("").classes(
                                "text-caption text-grey q-mt-n2"
                            )

                        with ui.tab_panel("Detect"):
                            with ui.row().classes(_IWN):
                                ui.label("Top").classes(_CAP)
                                _top_slider = ui.slider(
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=5,
                                    on_change=lambda e: _top_label.set_text(
                                        f"{int(e.value or 5)}"
                                    ),
                                ).classes(_GROW)
                                _top_slider.bind_value_to(_state, "top_labels")
                                _top_label = ui.label("5").classes("text-bold q-ml-sm")
                                ui.label("items").classes(_CAP)

                            with ui.row().classes(_IWN):
                                ui.label("Threshold").classes(_CAP)
                                ui.slider(
                                    min=config.CONFIDENCE_MIN,
                                    max=config.CONFIDENCE_MAX,
                                    step=config.CONFIDENCE_STEP,
                                    value=config.DEFAULT_CONFIDENCE,
                                    on_change=lambda e: (
                                        _state.set_confidence(e.value),
                                        _thresh_label.set_text(f"{e.value:.2f}"),
                                    ),
                                ).classes(_GROW)
                                _thresh_label = ui.label(
                                    f"{config.DEFAULT_CONFIDENCE:.2f}"
                                ).classes("text-bold q-ml-sm")

    # ---- Timer-driven refresh ------------------------------------------------
    def refresh_all():
        cam.set_source(f"/frame.jpg?{int(time.time() * 1000)}")
        _update_search_status()

    ui.timer(0.001, refresh_all)

    # ---- Page-local helper closures -----------------------------------------

    def _do_search():
        raw = (search_inp.value or "").strip()
        target = normalize_query(raw)
        if target:
            _state.submit_target(target)
            _search_status.set_text(f'Searching: "{target}"')
            search_inp.value = ""
            if _pipeline is not None:
                _pipeline.model.set_prompt(target)
        else:
            ui.notify("Enter what you want to find", type="warning")

    def _update_search_status():
        if _state.mode == "face":
            _search_status.set_text("Face Mesh Active")
        elif _state.submitted_target:
            _search_status.set_text(f'Searching: "{_state.submitted_target}"')
        else:
            _search_status.set_text("")


# ---- Lifecycle --------------------------------------------------------------


def _start_services():
    """Load models, create pipeline, start capture and inference."""
    global _pipeline, _device_str

    _device_str = get_device()
    logger.info("Starting services (device=%s)", _device_str)

    try:
        bundle = load_model_bundle(
            config.PROMPTED_MODEL, config.PROMPTFREE_MODEL, _state
        )
        _state.set_models_ready(True)
        _state.set_models_error(None)
        logger.info("Models loaded successfully")
    except Exception as exc:
        msg = f"Model loading failed: {exc}"
        logger.error(msg)
        _state.set_models_error(msg)
        return

    _pipeline = CapturePipeline(bundle, _state)
    _pipeline.start()

    # Startup timeout: if no FPS after 30s, log a warning
    def _check_startup():
        time.sleep(30)
        if not _state.camera_ready and not _state.camera_error:
            logger.warning("Camera not ready after 30s")
        if not _state.models_ready and not _state.models_error:
            logger.warning("Models not ready after 30s")

    threading.Thread(target=_check_startup, daemon=True).start()


def _stop_services():
    """Release camera and join worker threads."""
    logger.info("Shutting down services")
    if _pipeline is not None:
        _pipeline.stop()


napp.on_startup(lambda: threading.Thread(target=_start_services, daemon=True).start())
napp.on_shutdown(_stop_services)


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
