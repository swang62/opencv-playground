import logging
import os

# Suppress benign MediaPipe/TFLite C++ stderr logs (feedback manager, GL info, etc.)
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Redirect stderr during C++ library loading to suppress abseil/TFLite noise
_devnull = os.open(os.devnull, os.O_WRONLY)
_old_stderr = os.dup(2)
os.dup2(_devnull, 2)
os.close(_devnull)

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

from src.ui.app import main

os.dup2(_old_stderr, 2)
os.close(_old_stderr)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ in {"__main__", "__mp_main__"}:
    main()
