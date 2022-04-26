import sys

from loguru import logger


def set_log_verbosity(level: int) -> None:
    level = max(1, level)

    logger.enable("edgedroid")
    logger.remove()
    logger.add(sys.stderr, level=level)

    logger.info(f"Setting logging level to {level}")
