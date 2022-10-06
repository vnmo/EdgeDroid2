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
import itertools
import json
import pathlib
import socket
from typing import Any, Dict, Literal, Optional

import click
import yaml
from loguru import logger

from .client import StreamSocketEmulation
from ..common_cli import enable_logging


class JSONOption(click.ParamType):
    name = "JSON String"

    def convert(self, value, param, ctx) -> Dict:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            self.fail(f"{value!r} is not a valid JSON string.")


@click.command(
    "edgedroid-client",
    context_settings={"auto_envvar_prefix": "EDGEDROID_CLIENT"},
)
@click.argument(
    "host",
    type=str,
    envvar="EDGEDROID_CLIENT_HOST",
)
@click.argument(
    "port",
    type=click.IntRange(0, 65535),
    envvar="EDGEDROID_CLIENT_PORT",
)
@click.argument(
    "trace",
    type=str,
    envvar="EDGEDROID_CLIENT_TRACE",
    # default="square00",
    # show_default=True,
    # help="Name of the task trace to use for emulation.",
    # TODO: list traces? in tools!
)
@click.option(
    "--truncate",
    type=int,
    default=-1,
    help="Truncate the specified task trace to a given number of steps. "
    "Note that the server needs to be configured with the same value for the "
    "emulation to work.",
    show_default=False,
)
@click.option(
    "-m",
    "--timing-model",
    type=click.Choice(
        [
            "empirical",
            "theoretical",
            "constant",
            "naive",
            "fitted-naive",
        ],
        case_sensitive=True,
    ),
    default="theoretical",
    show_default=True,
    help="Execution time model to use:\n"
    "\n"
    "\b\n"
    "\t- 'empirical' samples directly from the underlying data.\n"
    "\t- 'theoretical' first fits distributions to the data and then samples.\n"
    "\t- 'constant' uses a constant execution time equal to the mean execution time "
    "of completely unimpaired samples in the underlying data.\n"
    "\t- 'naive' samples the underlying data without any grouping\n"
    "\t- 'fitted-naive' does the same as 'naive' but first fits a distribution to "
    "the data\n"
    "\t\n\n",
)
@click.option(
    "--timing-args",
    default="{}",
    show_default=False,
    type=JSONOption(),
    help="Arguments to pass on to the timing model, as a JSON string.",
)
@click.option(
    "-s",
    "--sampling-strategy",
    type=click.Choice(
        ["zero-wait", "ideal", "regular", "hold", "adaptive-aperiodic"],
        case_sensitive=False,
    ),
    default="zero-wait",
    show_default=True,
)
@click.option(
    "--sampling-args",
    default="{}",
    show_default=False,
    type=JSONOption(),
    help="Arguments to pass on to the sampling strategy, as a JSON string.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    default=None,
    show_default=True,
)
@click.option(
    "-v",
    "--verbose",
    type=bool,
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
    show_default=True,
)
@click.option(
    "--connect-timeout-seconds",
    "conn_tout",
    type=float,
    default=5.0,
    show_default=True,
    help="Time in seconds before the initial connection establishment times out.",
)
@click.option(
    "--max-connection-attempts",
    "max_attempts",
    type=int,
    default=5,
    show_default=True,
    help="Maximum connection retries, set to a 0 or a "
    "negative value for infinite retries.",
)
def edgedroid_client(
    host: str,
    port: int,
    trace: str,
    truncate: int,
    timing_model: Literal[
        "empirical",
        "theoretical",
        "constant",
        "naive",
        "fitted-naive",
    ],
    timing_args: Dict[str, Any],
    sampling_strategy: Literal[
        "zero-wait",
        "ideal",
        "regular",
        "hold",
        "adaptive-aperiodic",
    ],
    sampling_args: Dict[str, Any],
    verbose: bool,
    output_dir: Optional[pathlib.Path],
    conn_tout: float,
    max_attempts: int,
):
    """
    Run an EdgeDroid2 client.

    Connects to HOST:PORT and runs an emulation.
    """

    # prepare output paths
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        log_file = output_dir / "client.log"
        step_records_output = output_dir / "client.steps.csv"
        frame_records_output = output_dir / "client.frames.csv"
    else:
        log_file = None
        step_records_output = None
        frame_records_output = None

    enable_logging(verbose, log_file=log_file)

    # noinspection PyTypeChecker
    emulation = StreamSocketEmulation(
        trace=trace,
        model=timing_model.lower(),
        timing_args=timing_args,
        sampling=sampling_strategy.lower(),
        sampling_args=sampling_args,
        truncate=truncate,
    )

    logger.info(f"Connecting to remote server at {host}:{port}/tcp")
    try:
        if max_attempts <= 0:
            attempts = itertools.count(1)
            max_attpts_label = "inf"
        else:
            attempts = range(1, max_attempts + 1)
            max_attpts_label = f"{max_attempts:d}"

        for attempt in attempts:  # connection retry loop
            logger.debug(f"Connection attempt {attempt:d}/{max_attpts_label}")

            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(conn_tout)
                    sock.connect((host, port))
                    logger.success(f"Connected to {host}:{port}/tcp!")
                    sock.settimeout(None)  # no timeouts are needed
                    emulation.emulate(sock)
                    logger.success("Emulation finished")
                break  # success
            except socket.timeout:
                logger.warning("Connection timed out, retrying")
                continue
            except ConnectionRefusedError:
                logger.warning(f"{host}:{port} refused connection, retrying")
                continue
            except (socket.gaierror, socket.herror):
                logger.warning(f"Name lookup for target {host} failed, retrying")
                continue
            except socket.error as e:
                logger.critical(
                    f"Encountered unspecified socket error "
                    f"when connecting to {host}:{port}"
                )
                logger.exception(e)
                raise click.Abort()
            except Exception as e:
                # catch any other error and log it
                logger.exception(e)
                raise e
        else:
            # we only reach here if the code times out too many times!
            logger.critical("Reached maximum number of connection retries")
            logger.critical(f"Timed out connecting to backend at {host}:{port}")
            raise click.Abort()

    finally:
        # write outputs even if we abort above

        if step_records_output is not None:
            step_metrics = emulation.get_step_metrics()
            logger.info(f"Writing step metrics to {step_records_output}")
            step_metrics.to_csv(step_records_output)

        if frame_records_output is not None:
            frame_metrics = emulation.get_frame_metrics()
            logger.info(f"Writing frame metrics to {frame_records_output}")
            frame_metrics.to_csv(frame_records_output)

        if output_dir is not None:
            with (output_dir / "client.metadata.yml").open("w") as fp:
                yaml.safe_dump(
                    dict(
                        host=f"{host}:{port}",
                        task=trace,
                        sampling_strategy=sampling_strategy,
                        timing_model=timing_model,
                        timing_model_params=emulation.get_timing_model_parameters(),
                    ),
                    stream=fp,
                    explicit_start=True,
                    explicit_end=True,
                )
