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
import socket
import time
from collections import deque
from typing import Callable, Literal, Optional

import click
import numpy.typing as npt
import pandas as pd
from loguru import logger

from ..common import pack_frame, response_stream_unpack
from ... import data as e_data
from ...models import (
    AperiodicFrameSamplingModel,
    EdgeDroidModel,
    EmpiricalExecutionTimeModel,
    ExecutionTimeModel,
    HoldFrameSamplingModel,
    IdealFrameSamplingModel,
    ModelFrame,
    RegularFrameSamplingModel,
    TheoreticalExecutionTimeModel,
    ZeroWaitFrameSamplingModel,
)
from ...models.timings import NaiveExecutionTimeModel


class StreamSocketEmulation:
    def __init__(
        self,
        neuroticism: float,
        trace: str,
        fade_distance: int,
        model: Literal["theoretical", "empirical", "naive"] = "theoretical",
        sampling: str = "zero-wait",
        truncate: Optional[int] = None,
    ):

        trunc_log = f"(truncated to {truncate} steps)" if truncate is not None else ""
        logger.info(
            f"""
Initializing EdgeDroid model with:
- {neuroticism=:0.2f}
- {fade_distance=:d}
- {trace=} {trunc_log}
- {model=}
- {sampling=}
        """
        )

        # first thing first, prepare data
        frameset = e_data.load_default_trace(trace, truncate=truncate)

        match model:
            case "theoretical":
                timing_model: ExecutionTimeModel = (
                    TheoreticalExecutionTimeModel.from_default_data(
                        neuroticism=neuroticism,
                        transition_fade_distance=fade_distance,
                    )
                )
            case "empirical":
                timing_model: ExecutionTimeModel = (
                    EmpiricalExecutionTimeModel.from_default_data(
                        neuroticism=neuroticism,
                        transition_fade_distance=fade_distance,
                    )
                )
            case "naive":
                timing_model: ExecutionTimeModel = (
                    NaiveExecutionTimeModel.from_default_data()
                )
            case _:
                raise NotImplementedError(f"Unrecognized execution time model: {model}")

            # parse the sampling strategy
        sampling_vec = sampling.split("-")
        match sampling_vec:
            case ["zero", "wait"]:
                frame_model = ZeroWaitFrameSamplingModel(
                    e_data.load_default_frame_probabilities(),
                )
            case ["ideal"]:
                frame_model = IdealFrameSamplingModel(
                    e_data.load_default_frame_probabilities(),
                )
            case ["hold", time]:
                frame_model = HoldFrameSamplingModel(
                    e_data.load_default_frame_probabilities(),
                    hold_time_seconds=float(time),
                )
            case ["regular", time]:
                frame_model = RegularFrameSamplingModel(
                    e_data.load_default_frame_probabilities(),
                    sampling_interval_seconds=float(time),
                )
            case ["adaptive", "aperiodic"]:
                frame_model = AperiodicFrameSamplingModel(
                    e_data.load_default_frame_probabilities(),
                    execution_time_model=timing_model,
                )
            case _:
                raise NotImplementedError(f"No such sampling strategy: {sampling}")

        self._model = EdgeDroidModel(
            frame_trace=frameset, frame_model=frame_model, timing_model=timing_model
        )

        self._frame_records = deque()

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
                for frame_index, model_frame in enumerate(model_step):
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
                    transition, guidance_img, guidance_text, resp_size_bytes = next(
                        resp_stream
                    )
                    recv_time = time.monotonic()
                    rtt = recv_time - send_time
                    logger.debug("Received response from server")
                    logger.debug(f"Frame round-trip-time: {rtt:0.3f} seconds")
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
                fps = (frame_index + 1) / dt
                logger.debug(f"Step performance: {fps:0.2f} FPS")

        logger.warning("Emulation finished")
