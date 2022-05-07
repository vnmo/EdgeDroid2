#  Copyright (c) 2022 Manuel Olguín Muñoz <molguin@kth.se>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import pathlib
import sys
from typing import Optional

from loguru import logger


def enable_logging(verbose: bool, log_file: Optional[pathlib.Path] = None) -> None:
    logger.enable("edgedroid")
    logger.remove()

    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)
    if log_file is not None:
        logger.debug(f"Saving logs to {log_file}")
        logger.add(
            log_file,
            level=level,
            colorize=False,
            backtrace=True,
            diagnose=True,
            catch=True,
        )

    if verbose or log_file is not None:
        logger.warning(
            "Enabling verbose or file logging may affect application performance"
        )

    logger.info(f"Setting logging level to {level}")
