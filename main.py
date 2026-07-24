import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.app import main

if __name__ in {"__main__", "__mp_main__"}:
    main()
