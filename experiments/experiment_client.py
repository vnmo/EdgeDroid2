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

import contextlib
import itertools
import pathlib
import socket
import time
from collections import deque
from typing import Any, Callable, Dict, Optional

import click
import numpy.typing as npt
import pandas as pd
import yaml
from loguru import logger

from common import pack_frame, response_stream_unpack, enable_logging
from edgedroid.models import (
    EdgeDroidModel,
    ExecutionTimeModel,
    BaseFrameSamplingModel,
    FrameTimings,
    ModelFrame,
)
import edgedroid.data as e_data
from experiments import experiments


class NetworkEmulation:
    # noinspection PyDefaultArgument
    def __init__(
        self,
        trace: str,
        timing_model: ExecutionTimeModel,
        sampling_scheme: BaseFrameSamplingModel,
        truncate: int = -1,
    ):

        trunc_log = f"(truncated to {truncate} steps)" if truncate >= 0 else ""
        logger.info(
            f"""
Initializing EdgeDroid model with:
- Trace {trace} {trunc_log}
- Timing model: {timing_model.__class__.__name__}
- Sampling strategy: {sampling_scheme.__class__.__name__}
        """
        )

        self._model = EdgeDroidModel(
            frame_trace=e_data.load_default_trace(trace, truncate=truncate),
            frame_model=sampling_scheme,
            timing_model=timing_model,
        )

        self._frame_records = deque()

    def get_timing_model_parameters(self) -> Dict[str, Any]:
        return self._model.timing_model_params()

    def get_step_metrics(self) -> pd.DataFrame:
        return self._model.model_step_metrics()

    def get_frame_metrics(self) -> pd.DataFrame:
        return pd.DataFrame(self._frame_records).set_index("seq")

    def emulate(
        self,
        sock: socket.SocketType,
        emit_cb: Callable[[ModelFrame], None] = lambda _: None,
        resp_cb: Callable[[bool, npt.NDArray, str], None] = lambda t, i, s: None,
    ) -> None:
        """
        # TODO: document

        Parameters
        ----------
        sock
        emit_cb
        resp_cb

        Returns
        -------

        """

        logger.warning("Starting emulation")
        start_time = time.time()
        start_time_mono = time.monotonic()
        with contextlib.closing(response_stream_unpack(sock)) as resp_stream:
            for step_num, model_step in enumerate(self._model.play_steps()):
                logger.info(f"Current step: {step_num}")
                ti = time.monotonic()
                frame_timings: Optional[FrameTimings] = None
                while True:
                    try:
                        model_frame = model_step.send(frame_timings)
                    except StopIteration:
                        break
                    # package and send the frame
                    logger.debug(
                        f"Sending frame:\n"
                        f"\tSeq: {model_frame.seq}\n"
                        f"\tTag: {model_frame.frame_tag}\n"
                        f"\tStep index: {model_frame.step_index}\n"
                        f"\tFrame step seq: {model_frame.step_seq}"
                    )
                    payload = pack_frame(model_frame.seq, model_frame.frame_data)
                    send_time = time.monotonic()
                    sock.sendall(payload)
                    emit_cb(model_frame)

                    # wait for response
                    logger.debug("Waiting for response from server")
                    (
                        transition,
                        processing_time_s,
                        guidance_img,
                        guidance_text,
                        resp_size_bytes,
                    ) = next(resp_stream)

                    recv_time = time.monotonic()
                    rtt = recv_time - send_time
                    nettime_s = rtt - processing_time_s

                    frame_timings = FrameTimings(nettime_s, processing_time_s)

                    logger.debug("Received response from server")
                    logger.debug(
                        f"Timing metrics: "
                        f"RTT {rtt:0.3f}s | "
                        f"Proc. time {processing_time_s:0.3f}s | "
                        f"Net. time {nettime_s:0.3f}s"
                    )
                    logger.info(f"Guidance: {guidance_text}")
                    resp_cb(transition, guidance_img, guidance_text)

                    # log the frame

                    self._frame_records.append(
                        {
                            "seq": model_frame.seq,
                            "step_index": model_frame.step_index,
                            "step_seq": model_frame.step_seq,
                            "expected_tag": model_frame.frame_tag,
                            "transition": transition,
                            "send_time": start_time + (send_time - start_time_mono),
                            "rtt": rtt,
                            "send_size_bytes": len(payload),
                            "recv_size_bytes": resp_size_bytes,
                            "frame_step_time": model_frame.step_frame_time,
                            **{
                                f"extra_{k}": v
                                for k, v in model_frame.extra_data.items()
                            },
                        }
                    )

                # when step iterator finishes, we should have reached a success frame!
                if not transition:
                    logger.error(
                        "Expected step transition, "
                        "but backend returned non-success response."
                    )
                    raise click.Abort()

                logger.success("Advancing to next step")
                dt = time.monotonic() - ti
                fps = model_frame.step_seq / dt
                logger.debug(f"Step performance: {fps:0.2f} FPS")

        logger.warning("Emulation finished")


@click.command(
    "edgedroid-experiment-client",
    context_settings={"auto_envvar_prefix": "EDGEDROID_CLIENT"},
)
@click.argument(
    "experiment-id",
    type=click.Choice(
        list(experiments.keys()),
        case_sensitive=True,
    ),
    envvar="EDGEDROID_CLIENT_EXPERIMENT_ID",
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
    type=click.Choice(list(e_data.load._default_traces), case_sensitive=True),
    envvar="EDGEDROID_CLIENT_TRACE",
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
    experiment_id: str,
    host: str,
    port: int,
    trace: str,
    truncate: int,
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

    timing_model, sampling_scheme, metadata = experiments[experiment_id]()

    # noinspection PyTypeChecker
    emulation = NetworkEmulation(
        trace=trace,
        timing_model=timing_model,
        sampling_scheme=sampling_scheme,
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
                        truncate=truncate,
                        **metadata,
                    ),
                    stream=fp,
                    explicit_start=True,
                    explicit_end=True,
                )


if __name__ == "__main__":
    edgedroid_client()
