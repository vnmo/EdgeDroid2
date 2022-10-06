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
from typing import Any, Callable, Dict, Literal, Type

import click
import numpy.typing as npt
import pandas as pd
from loguru import logger

from ..common import pack_frame, response_stream_unpack
from ... import data as e_data
from ...models import (
    AperiodicFrameSamplingModel,
    BaseFrameSamplingModel,
    EdgeDroidModel,
    EmpiricalExecutionTimeModel,
    ExecutionTimeModel,
    HoldFrameSamplingModel,
    IdealFrameSamplingModel,
    ModelFrame,
    RegularFrameSamplingModel,
    TheoreticalExecutionTimeModel,
    ZeroWaitFrameSamplingModel,
    NaiveExecutionTimeModel,
    ConstantExecutionTimeModel,
    FittedNaiveExecutionTimeModel,
)
from ...models.sampling import TBaseSampling


def _get_timing_model_cls(model_name: str) -> Type[ExecutionTimeModel]:
    try:
        timing_model_cls: Type[ExecutionTimeModel] = {
            "theoretical": TheoreticalExecutionTimeModel,
            "empirical": EmpiricalExecutionTimeModel,
            "constant": ConstantExecutionTimeModel,
            "naive": NaiveExecutionTimeModel,
            "fitted-naive": FittedNaiveExecutionTimeModel,
        }[model_name]
        return timing_model_cls
    except KeyError as e:
        raise NotImplementedError(
            f"Unrecognized execution time model: {model_name}"
        ) from e


class AdaptiveAperiodicSampling(AperiodicFrameSamplingModel):
    # noinspection PyDefaultArgument
    @classmethod
    def from_default_data(
        cls: Type[TBaseSampling],
        execution_time_model: str,
        delay_cost_window: int = 5,
        beta: float = 1.0,
        init_nettime_guess=0.3,
        proctime: float = 0.0,
        timing_model_params: Dict[str, Any] = {"neuroticism": None},
        *args,
        **kwargs,
    ) -> TBaseSampling:
        probs = e_data.load_default_frame_probabilities()
        timing_model_cls = _get_timing_model_cls(execution_time_model)
        return cls(
            probabilities=probs,
            execution_time_model=timing_model_cls.from_default_data(
                **timing_model_params
            ),
            success_tag="success",
            delay_cost_window=delay_cost_window,
            beta=beta,
            init_nettime_guess=init_nettime_guess,
            proctime=proctime,
        )


class StreamSocketEmulation:
    # noinspection PyDefaultArgument
    def __init__(
        self,
        trace: str,
        model: Literal[
            "theoretical",
            "empirical",
            "constant",
            "naive",
            "fitted-naive",
        ] = "theoretical",
        sampling: Literal[
            "zero-wait",
            "ideal",
            "regular",
            "hold",
            "adaptive-aperiodic",
        ] = "zero-wait",
        truncate: int = -1,
        timing_args: Dict[str, Any] = {},
        sampling_args: Dict[str, Any] = {},
    ):

        trunc_log = f"(truncated to {truncate} steps)" if truncate >= 0 else ""
        logger.info(
            f"""
Initializing EdgeDroid model with:
- Trace {trace} {trunc_log}
- Timing model: {model} (args: {timing_args})
- Sampling strategy: {sampling} (args: {sampling_args})
        """
        )

        # first thing first, prepare data
        frameset = e_data.load_default_trace(trace, truncate=truncate)

        timing_model_cls = _get_timing_model_cls(model)
        timing_model = timing_model_cls.from_default_data(**timing_args)

        try:
            sampling_cls: Type[BaseFrameSamplingModel] = {
                "zero-wait": ZeroWaitFrameSamplingModel,
                "ideal": IdealFrameSamplingModel,
                "hold": HoldFrameSamplingModel,
                "regular": RegularFrameSamplingModel,
                "adaptive-aperiodic": AdaptiveAperiodicSampling,
            }[sampling]
        except KeyError as e:
            raise NotImplementedError(f"No such sampling strategy: {sampling}") from e

        frame_model = sampling_cls.from_default_data(**sampling_args)

        self._model = EdgeDroidModel(
            frame_trace=frameset, frame_model=frame_model, timing_model=timing_model
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
