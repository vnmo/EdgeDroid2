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
from typing import Callable, Literal

import click
import pandas as pd
from loguru import logger

from ..common import response_stream_unpack, pack_frame
from ... import data as e_data
from ...models import (
    EdgeDroidModel,
    EmpiricalExecutionTimeModel,
    ExecutionTimeModel,
    FrameModel,
    ModelFrame,
    TheoreticalExecutionTimeModel,
)

import numpy.typing as npt


class StreamSocketEmulation:
    def __init__(
        self,
        neuroticism: float,
        trace: str,
        fade_distance: int,
        model: Literal["theoretical", "empirical"] = "theoretical",
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

        # prepare models
        if model == "theoretical":
            timing_model: ExecutionTimeModel = (
                TheoreticalExecutionTimeModel.from_default_data(
                    neuroticism=neuroticism,
                    transition_fade_distance=fade_distance,
                )
            )
        else:
            timing_model: ExecutionTimeModel = (
                EmpiricalExecutionTimeModel.from_default_data(
                    neuroticism=neuroticism,
                    transition_fade_distance=fade_distance,
                )
            )

        frame_model = FrameModel(e_data.load_default_frame_probabilities())

        self._model = EdgeDroidModel(
            frame_trace=frameset, frame_model=frame_model, timing_model=timing_model
        )

    def get_step_metrics(self) -> pd.DataFrame:
        return self._model.model_step_metrics()

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
                    sock.sendall(payload)
                    emit_cb(model_frame)

                    # wait for response
                    logger.debug("Waiting for response from server")
                    transition, guidance_img, guidance_text = next(resp_stream)
                    logger.debug("Received response from server")
                    logger.info(f"Guidance: {guidance_text}")
                    resp_cb(transition, guidance_img, guidance_text)

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
