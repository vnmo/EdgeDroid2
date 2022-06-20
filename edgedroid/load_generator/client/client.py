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
from typing import Callable, Literal

import click
import pandas as pd
from loguru import logger

from ..common import response_stream_unpack, pack_frame
from ... import data as e_data
from ...models import (
    BaseFrameSamplingModel,
    EdgeDroidModel,
    EmpiricalExecutionTimeModel,
    ExecutionTimeModel,
    IdealFrameSamplingModel,
    ZeroWaitFrameSamplingModel,
    ModelFrame,
    TheoreticalExecutionTimeModel,
)

import numpy.typing as npt

from ...models.timings import NaiveExecutionTimeModel


class StreamSocketEmulation:
    def __init__(
        self,
        neuroticism: float,
        trace: str,
        fade_distance: int,
        model: Literal["theoretical", "empirical", "naive"] = "theoretical",
        sampling: Literal["zero-wait", "ideal"] = "zero-wait",
    ):
        logger.info(
            f"Initializing EdgeDroid model with neuroticism {neuroticism:0.2f} and "
            f"fade distance {fade_distance:d} steps"
        )
        logger.info(f"Model type: {model}")
        logger.info(f"Trace: {trace}")

        # should be able to use a single thread for everything

        # first thing first, prepare data
        frameset = e_data.load_default_trace(trace)

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

        match sampling:
            case "zero-wait":
                sampling_cls = ZeroWaitFrameSamplingModel
            case "ideal":
                sampling_cls = IdealFrameSamplingModel
            case _:
                raise NotImplementedError(f"No such sampling strategy: {sampling}")

        frame_model: BaseFrameSamplingModel = sampling_cls(
            e_data.load_default_frame_probabilities()
        )

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
                for model_frame in model_step:
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

                    if model_frame.frame_tag in ("success", "initial"):
                        if transition:
                            # if we receive a response for a success frame, advance the
                            # model
                            logger.success("Advancing to next step")
                            break
                        else:
                            logger.error(
                                "Received unexpected unsuccessful response from "
                                "server, aborting"
                            )
                            raise click.Abort()

        logger.warning("Emulation finished")
