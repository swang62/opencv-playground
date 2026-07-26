"""Native OS file dialog using tkinter — works on macOS, Windows, and Linux.

Returns the local file path directly; no upload, no system binary calls.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import filedialog

logger = logging.getLogger(__name__)

VIDEO_TYPES = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.wmv *.flv *.m4v *.mpeg"),
    ("All files", "*.*"),
]


def pick_video() -> str | None:
    """Open a native file dialog for video selection.

    Returns the absolute path to the selected file, or ``None`` if the
    user cancelled.
    """
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=VIDEO_TYPES,
        )
    except Exception as exc:
        logger.warning("File picker failed: %s", exc)
        return None
    finally:
        root.destroy()
    return path if path else None
