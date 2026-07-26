"""Native OS file picker dialog — no file upload, returns the local path.

Uses ``osascript`` on macOS, falls back to ``zenity`` or ``kdialog`` on
Linux, and ``pythoncom`` + ``win32com.client`` on Windows.
"""

from __future__ import annotations

import logging
import platform
import subprocess

logger = logging.getLogger(__name__)

SYSTEM = platform.system()


def pick_video() -> str | None:
    """Open a native file dialog for video selection.

    Returns the absolute path to the selected file, or ``None`` if the
    user cancelled.
    """
    if SYSTEM == "Darwin":
        return _pick_macos()
    elif SYSTEM == "Linux":
        return _pick_linux()
    elif SYSTEM == "Windows":
        return _pick_windows()
    else:
        logger.warning("Unsupported platform: %s", SYSTEM)
        return None


def _pick_macos() -> str | None:
    script = """
    set videoTypes to {"public.movie", "public.mpeg", "public.avi", "public.3gpp", "public.mpeg-4"}
    set filePath to choose file with prompt "Select a video file" of type videoTypes
    return POSIX path of filePath
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        path = result.stdout.strip()
        return path if path else None
    except Exception as exc:
        logger.warning("macOS file picker failed: %s", exc)
        return None


def _pick_linux() -> str | None:
    for cmd in ("zenity", "kdialog"):
        try:
            if cmd == "zenity":
                args = ["zenity", "--file-selection", "--title=Select a video file"]
            else:
                args = [
                    "kdialog",
                    "--getopenfilename",
                    ".",
                    "*.mp4 *.avi *.mov *.mkv *.webm *.wmv *.flv *.m4v",
                ]
            result = subprocess.run(args, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            continue
    logger.warning("No supported file picker found (try zenity or kdialog)")
    return None


def _pick_windows() -> str | None:
    try:
        import pythoncom  # pyright: ignore[reportMissingModuleSource]
        import win32com.client  # pyright: ignore[reportMissingModuleSource]

        pythoncom.CoInitialize()
        dialog = win32com.client.Dispatch("UserAccounts.CommonDialog")
        dialog.Filter = "Video Files|*.mp4;*.avi;*.mov;*.mkv;*.webm;*.wmv;*.flv;*.m4v"
        dialog.FilterIndex = 1
        dialog.ShowOpen()
        return dialog.FileName or None
    except Exception as exc:
        logger.warning("Windows file picker failed: %s", exc)
        return None
