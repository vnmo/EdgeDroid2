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
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, Generator, Iterator, List, Optional

import numpy.typing as npt
import pandas as pd

from .sampling import (
    AperiodicFrameSamplingModel,
    AperiodicPowerFrameSamplingModel,
    BaseFrameSamplingModel,
    FrameSample,
    FrameSet,
    FrameTimings,
    HoldFrameSamplingModel,
    IdealFrameSamplingModel,
    RegularFrameSamplingModel,
    ZeroWaitFrameSamplingModel,
)
from .timings import (
    ConstantExecutionTimeModel,
    EmpiricalExecutionTimeModel,
    ExecutionTimeModel,
    FittedNaiveExecutionTimeModel,
    NaiveExecutionTimeModel,
    TheoreticalExecutionTimeModel,
    preprocess_data,
    ExpKernelRollingTTFETModel,
    DistExpKernelRollingTTFETModel,
)

__all__ = [
    "ExecutionTimeModel",
    "TheoreticalExecutionTimeModel",
    "EmpiricalExecutionTimeModel",
    "FrameSet",
    "ZeroWaitFrameSamplingModel",
    "ModelFrame",
    "EdgeDroidModel",
    "preprocess_data",
    "IdealFrameSamplingModel",
    "BaseFrameSamplingModel",
    "HoldFrameSamplingModel",
    "RegularFrameSamplingModel",
    "AperiodicFrameSamplingModel",
    "AperiodicPowerFrameSamplingModel",
    "ConstantExecutionTimeModel",
    "NaiveExecutionTimeModel",
    "FittedNaiveExecutionTimeModel",
    "FrameTimings",
    "ExpKernelRollingTTFETModel",
    "DistExpKernelRollingTTFETModel",
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
    extra_data: Dict[str, Any]


@dataclass(frozen=True, eq=True)
class StepRecord:
    step_number: int
    # impairment_score: float
    step_start: float
    step_start_monotonic: float
    step_end: float
    step_end_monotonic: float
    first_frame_monotonic: float
    last_frame_monotonic: float
    last_frame_rtt: float
    execution_time: float
    step_duration: float
    time_to_feedback: float  # difference between step duration and execution time
    wait_time: float  # time between execution time and when final sample is taken
    frame_count: int

    def to_dict(self) -> Dict[str, int | float]:
        return asdict(self)


class EdgeDroidModel:
    """
    Implements the full end-to-end emulation of a human user in Cognitive
    Assistance.
    """

    def __init__(
        self,
        frame_trace: FrameSet,
        timing_model: ExecutionTimeModel,
        frame_model: BaseFrameSamplingModel,
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
            A BaseFrameSamplingModel object to provide frame distribution information at
            runtime.
        """
        super(EdgeDroidModel, self).__init__()

        self._timings = timing_model
        self._frames = frame_trace
        self._frame_dists = frame_model
        self._frame_count = 0
        self._step_records: List[StepRecord] = []

    def reset(self) -> None:
        """
        Resets this model.
        """
        self._step_records.clear()
        self._timings.reset()
        self._frame_count = 0

    def model_step_metrics(self) -> pd.DataFrame:
        return pd.DataFrame(
            [a.to_dict() for a in self._step_records],
        ).set_index("step_number")

    def timing_model_params(self) -> Dict[str, Any]:
        return self._timings.get_model_params()

    @property
    def step_count(self) -> int:
        return self._frames.step_count

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def play_steps(
        self,
    ) -> Iterator[Generator[ModelFrame, FrameTimings, None]]:
        """
        TODO: document
        """
        task_start, task_start_mono = time.time(), time.monotonic()
        prev_step_end = task_start_mono

        self.reset()
        step_frame_timestamps = deque()
        initial_timings = {}

        def _init_iter() -> Generator[ModelFrame, FrameTimings, None]:
            self._frame_count += 1
            step_frame_timestamps.append(time.monotonic())
            nettime, proctime = yield ModelFrame(
                seq=self._frame_count,
                step_seq=1,
                step_index=-1,
                step_target_time=0,
                step_frame_time=0,
                frame_tag="initial",
                frame_data=self._frames.get_initial_frame(),
                extra_data={},
            )
            initial_timings["nettime"] = nettime
            initial_timings["proctime"] = proctime

        yield _init_iter()
        dt = time.monotonic() - prev_step_end

        step_record = StepRecord(
            step_number=0,
            step_start=task_start,
            step_start_monotonic=prev_step_end,
            step_end=task_start + dt,
            step_end_monotonic=prev_step_end + dt,
            first_frame_monotonic=step_frame_timestamps[0],
            last_frame_monotonic=step_frame_timestamps[-1],
            last_frame_rtt=(prev_step_end + dt) - step_frame_timestamps[-1],
            execution_time=0.0,
            step_duration=dt,
            time_to_feedback=dt,
            wait_time=step_frame_timestamps[-1] - task_start_mono,
            frame_count=len(step_frame_timestamps),
        )

        self._step_records.append(step_record)
        prev_step_end = step_record.step_end_monotonic

        # start the actual sampling scheme
        self._frame_dists.update_timings(
            [initial_timings["nettime"]], [initial_timings["proctime"]]
        )

        for step_index in range(self.step_count):
            # get a step duration
            ttf = self._step_records[-1].time_to_feedback
            execution_time = self._timings.advance(ttf).get_execution_time()

            # clear the frame timestamp buffer
            step_frame_timestamps.clear()

            def _frame_iter_for_step() -> Generator[ModelFrame, FrameTimings, None]:
                # TODO: implement sampling records
                # replay frames for step
                frame_iter = self._frame_dists.step_iterator(
                    target_time=execution_time,
                    ttf=ttf,
                )
                frame_timings: Optional[FrameTimings] = None

                while True:
                    try:
                        sample = frame_iter.send(frame_timings)
                    except StopIteration:
                        break

                    self._frame_count += 1
                    # record frame emission timestamp
                    step_frame_timestamps.append(time.monotonic())
                    nettime, proctime = yield ModelFrame(
                        seq=self._frame_count,
                        step_seq=sample.seq,
                        step_index=step_index,
                        step_frame_time=sample.instant,
                        step_target_time=execution_time,
                        frame_tag=sample.sample_tag,
                        frame_data=self._frames.get_frame(
                            step_index,
                            sample.sample_tag,
                        ),
                        extra_data=sample.extra,
                    )
                    frame_timings = FrameTimings(nettime, proctime)

            yield _frame_iter_for_step()
            dt = time.monotonic() - prev_step_end  # duration of step

            step_start = task_start + (prev_step_end - task_start_mono)
            # TODO: this is a lot of processing... push to another process?
            step_record = StepRecord(
                step_number=step_index + 1,
                step_start=step_start,
                step_start_monotonic=prev_step_end,
                step_end=step_start + dt,
                step_end_monotonic=prev_step_end + dt,
                first_frame_monotonic=step_frame_timestamps[0],
                last_frame_monotonic=step_frame_timestamps[-1],
                last_frame_rtt=(prev_step_end + dt) - step_frame_timestamps[-1],
                execution_time=execution_time,
                step_duration=dt,
                time_to_feedback=dt - execution_time,
                wait_time=step_frame_timestamps[-1] - (prev_step_end + execution_time),
                frame_count=len(step_frame_timestamps),
            )

            self._step_records.append(step_record)
            prev_step_end = step_record.step_end_monotonic  # update checkpoint
