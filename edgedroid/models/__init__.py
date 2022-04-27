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

import time
from dataclasses import dataclass
from typing import Iterator

import numpy.typing as npt

from .frames import FrameModel, FrameSet
from .timings import (
    EmpiricalExecutionTimeModel,
    ExecutionTimeModel,
    TheoreticalExecutionTimeModel,
    preprocess_data,
)

__all__ = [
    "ExecutionTimeModel",
    "TheoreticalExecutionTimeModel",
    "EmpiricalExecutionTimeModel",
    "FrameSet",
    "FrameModel",
    "ModelFrame",
    "EdgeDroidModel",
    "preprocess_data",
]


@dataclass(frozen=True)
class ModelFrame:
    seq: int  # absolute seq number
    step_seq: int  # seq number for current step
    step_index: int
    step_frame_time: float
    step_target_time: float
    frame_tag: str
    frame_data: npt.NDArray


class EdgeDroidModel:
    """
    Implements the full end-to-end emulation of a human user in Cognitive
    Assistance.
    """

    def __init__(
        self,
        frame_trace: FrameSet,
        timing_model: ExecutionTimeModel,
        frame_model: FrameModel,
    ):
        """
        Parameters
        ----------
        frame_trace
            A FrameSet object containing the video frame trace for the
            target task.
        timing_model
            An ExecutionTimeModel object to provide the timing information.
        frame_model
            A FrameModel object to provide frame distribution information at
            runtime.
        """
        super(EdgeDroidModel, self).__init__()

        self._timings = timing_model
        self._frames = frame_trace
        self._frame_dists = frame_model
        self._frame_count = 0

    @property
    def step_count(self) -> int:
        return self._frames.step_count

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def play_steps(self) -> Iterator[Iterator[ModelFrame]]:
        """
        TODO: document

        An iterator for iterators
        Returns
        -------

        """

        """
                Run this model.

                This function returns an iterator yielding video frames for each
                instant in the emulation. Example usage::

                    model = EdgeDroidModel(...)
                    for frame in model.play():
                        result = send_frame_to_backend(frame)
                        ...


                This iterator maintains an internal state of the task and
                automatically produces realistic frames and timings.

                Returns
                -------
                Iterator
                    An Iterator that yields appropriate video frames as numpy arrays.
                """

        def _init_iter() -> Iterator[ModelFrame]:
            while True:
                self._frame_count += 1
                yield ModelFrame(
                    seq=self._frame_count,
                    step_seq=1,
                    step_index=-1,
                    step_target_time=0,
                    step_frame_time=0,
                    frame_tag="initial",
                    frame_data=self._frames.get_initial_frame(),
                )

        yield _init_iter()
        prev_step_end = time.monotonic()

        for step_index in range(self.step_count):
            # get a step duration
            delay = time.monotonic() - prev_step_end
            self._timings.set_delay(delay)
            step_duration = self._timings.get_execution_time()

            def _frame_iter_for_step() -> Iterator[ModelFrame]:
                # replay frames for step
                for seq, (frame_tag, instant) in enumerate(
                    self._frame_dists.step_iterator(
                        target_time=step_duration, infinite=True
                    )
                ):
                    self._frame_count += 1
                    yield ModelFrame(
                        seq=self._frame_count,
                        step_seq=seq + 1,
                        step_index=step_index,
                        step_frame_time=instant,
                        step_target_time=step_duration,
                        frame_tag=frame_tag,
                        frame_data=self._frames.get_frame(step_index, frame_tag),
                    )

            yield _frame_iter_for_step()
            prev_step_end = time.monotonic()
