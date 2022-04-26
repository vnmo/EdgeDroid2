import sys

from loguru import logger


def enable_logging(verbose: bool) -> None:
    logger.enable("edgedroid")
    logger.remove()

    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)
    logger.info(f"Setting logging level to {level}")
