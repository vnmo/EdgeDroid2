import contextlib
import time
from dataclasses import dataclass
from typing import Iterator

import numpy.typing as npt

from .timings import (
    ExecutionTimeModel,
    TheoreticalExecutionTimeModel,
    EmpiricalExecutionTimeModel,
    preprocess_data,
)
from .frames import FrameModel, FrameSet

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
        self._advance_step = False

    def advance_step(self) -> None:
        self._advance_step = True

    @property
    def step_count(self) -> int:
        return self._frames.step_count

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def play(self) -> Iterator[ModelFrame]:
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

        self._advance_step = False

        self._timings.reset()

        prev_step_end = time.monotonic()

        # yield the initial frame of the task
        prev_success = self._frames.get_initial_frame()

        while not self._advance_step:
            self._frame_count += 1
            yield ModelFrame(
                seq=self._frame_count,
                step_seq=1,
                step_index=-1,
                step_target_time=0,
                step_frame_time=0,
                frame_tag="initial",
                frame_data=prev_success,
            )

        # task is now running
        for step_index in range(self._frames.step_count):
            # get a step duration
            delay = time.monotonic() - prev_step_end
            self._timings.set_delay(delay)
            step_duration = self._timings.get_execution_time()
            self._advance_step = False

            # replay frames for step
            seq = 0
            with contextlib.closing(
                self._frame_dists.step_iterator(
                    target_time=step_duration, infinite=True
                )
            ) as step_frames:
                while not self._advance_step:
                    frame_tag, instant = next(step_frames)
                    seq += 1
                    self._frame_count += 1

                    # FIXME: hardcoded string tags
                    if frame_tag == "repeat":
                        yield ModelFrame(
                            seq=self._frame_count,
                            step_seq=seq,
                            step_index=step_index,
                            step_frame_time=instant,
                            step_target_time=step_duration,
                            frame_tag=frame_tag,
                            frame_data=prev_success,
                        )
                    elif frame_tag == "success":
                        prev_success = self._frames.get_frame(step_index, frame_tag)

                        prev_step_end = time.monotonic()
                        yield ModelFrame(
                            seq=self._frame_count,
                            step_seq=seq,
                            step_index=step_index,
                            step_frame_time=instant,
                            step_target_time=step_duration,
                            frame_tag=frame_tag,
                            frame_data=prev_success,
                        )
                    else:
                        yield ModelFrame(
                            seq=self._frame_count,
                            step_seq=seq,
                            step_index=step_index,
                            step_frame_time=instant,
                            step_target_time=step_duration,
                            frame_tag=frame_tag,
                            frame_data=self._frames.get_frame(step_index, frame_tag),
                        )