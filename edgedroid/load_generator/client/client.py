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
from loguru import logger

from ..common import EdgeDroidFrame, response_stream_unpack
from ... import data as e_data
from ...models import (
    EdgeDroidModel,
    EmpiricalExecutionTimeModel,
    ExecutionTimeModel,
    FrameModel,
    ModelFrame,
    TheoreticalExecutionTimeModel,
)


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

    def emulate(
        self,
        sock: socket.SocketType,
        emit_cb: Callable[[ModelFrame], None] = lambda _: None,
        resp_cb: Callable[[bool], None] = lambda _: None,
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
            step = 0
            logger.info(f"Current step: {step}")
            for model_frame in self._model.play():
                # package and send the frame
                logger.debug(
                    f"Sending frame:\n"
                    f"\tSeq: {model_frame.seq}\n"
                    f"\tTag: {model_frame.frame_tag}\n"
                    f"\tStep index: {model_frame.step_index}\n"
                    f"\tFrame step seq: {model_frame.step_seq}"
                )
                payload = EdgeDroidFrame(model_frame.seq, model_frame.frame_data).pack()
                sock.sendall(payload)
                emit_cb(model_frame)

                # wait for response
                logger.debug("Waiting for response from server")
                resp = next(resp_stream)
                logger.debug("Received response from server")
                resp_cb(resp)

                if model_frame.frame_tag in ("success", "initial"):
                    if resp:
                        # if we receive a response for a success frame, advance the
                        # model
                        logger.info("Advancing to next step")
                        self._model.advance_step()
                        step += 1
                        logger.info(f"Current step: {step}")
                    else:
                        logger.error(
                            "Received unexpected unsuccessful response from server, "
                            "aborting"
                        )
                        raise click.Abort()

        logger.warning("Emulation finished")
