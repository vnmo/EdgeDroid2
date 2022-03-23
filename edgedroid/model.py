import time
from dataclasses import dataclass
from typing import Iterator

import nptyping as npt

from .execution_times import ExecutionTimeModel
from .frames import FrameModel, FrameSet


@dataclass(frozen=True)
class ModelFrame:
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

    @property
    def step_count(self) -> int:
        return self._frames.step_count

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

        self._timings.reset()

        prev_step_end = time.monotonic()

        # yield the initial frame of the task
        prev_success = self._frames.get_initial_frame()
        yield ModelFrame(
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

            # replay frames for step
            for frame_tag, instant in self._frame_dists.step_iterator(
                target_time=step_duration
            ):
                # FIXME: hardcoded string tags

                if frame_tag == "repeat":
                    yield ModelFrame(
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
                        step_index=step_index,
                        step_frame_time=instant,
                        step_target_time=step_duration,
                        frame_tag=frame_tag,
                        frame_data=prev_success,
                    )
                else:
                    yield ModelFrame(
                        step_index=step_index,
                        step_frame_time=instant,
                        step_target_time=step_duration,
                        frame_tag=frame_tag,
                        frame_data=self._frames.get_frame(step_index, frame_tag),
                    )
