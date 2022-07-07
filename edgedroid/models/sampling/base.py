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

from __future__ import annotations

import abc
import time
from collections import deque
from os import PathLike
from typing import Any, Dict, Iterator, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml


class FrameSet:
    """
    Abstraction of a set of video frames for the model.

    Easiest way to build an instance of this class is using the
    FrameSet.from_datafile class method. This method takes a tracefile in
    .npz format and parses it.
    """

    def __init__(
        self,
        name: str,
        initial_frame: npt.NDArray,
        steps: Sequence[Dict[str, npt.NDArray]],
    ):
        """
        Parameters
        ----------
        name
            A name for the task represented by this trace of frames.
        initial_frame
            The initial video frame required by the backend to initialize
            the task.
        steps
            A sequence containing dictionaries mapping frame tags to video
            frames, in order of steps.
            Note that it is expected that all steps have the same tags!
        """

        self._name = name
        self._init_frame = initial_frame
        self._steps = tuple(steps)
        self._num_steps = len(self._steps)

    def __str__(self) -> str:
        return yaml.safe_dump(
            {
                "name": self._name,
                "num_steps": self.step_count,
                "initial_frame": f"{len(self._init_frame.tobytes())} bytes",
                "steps": [
                    {tag: f"{len(data.tobytes())} bytes" for tag, data in step.items()}
                    for step in self._steps
                ],
            }
        )

    @property
    def step_count(self) -> int:
        return self._num_steps

    @property
    def name(self) -> str:
        return self._name

    def get_initial_frame(self) -> npt.NDArray:
        """
        Returns
        -------
        npt.NDArray
            The initial video frame for this task.
        """
        return self._init_frame.copy()

    def get_frame(self, step_index: Any, frame_tag: str) -> npt.NDArray:
        """
        Looks up a frame for a specific tag in a step.

        Parameters
        ----------
        step_index
            Step index.
        frame_tag
            Frame tag to look up.

        Returns
        -------
        npt.NDArray
            A video frame.
        """
        return self._steps[step_index][frame_tag].copy()

    @classmethod
    def from_datafile(cls, task_name: str, trace_path: PathLike | str) -> FrameSet:
        """
        Opens a frame tracefile and parses it.

        Traces correspond to compressed numpy array files (.npz) containing
        the following arrays:

            - An array called "initial" corresponding to the initial frame for
              the task.
            - A number `M x N` of arrays, where M is the number of different
              possible tags for frames during a step, and N corresponds to
              the number of steps in the tag. Each of these arrays is named
              following the convention "step_<step index (two digits,
              0-padded)>_<frame tag>".

        Parameters
        ----------
        task_name
            Task name for this trace.
        trace_path
            Path to the datafile.

        Returns
        -------
        FrameSet
            A FrameSet object.
        """

        data = np.load(trace_path)

        # trace NPZ file contains initial frame + 3 frames per step
        # success, blank, and low_confidence
        # TODO: this assumes 3 frame categories per step (success, low confidence and
        #  blank (repeat is simply the previous success)). Maybe we should add a way
        #  of configuring that.
        assert (len(data) - 1) % 3 == 0
        num_steps = (len(data) - 1) // 3

        init_frame = data["initial"]
        # TODO: hardcoded categories
        steps = deque()
        repeat = init_frame

        for step in range(num_steps):
            step_dict = {}
            for tag in ("success", "blank", "low_confidence"):
                step_dict[tag] = data[f"step{step:02d}_{tag}"]

            step_dict["repeat"] = repeat
            repeat = step_dict["success"]
            steps.append(step_dict)

        return FrameSet(name=task_name, initial_frame=init_frame, steps=steps)


class BaseFrameSamplingModel(abc.ABC):
    def __init__(self, probabilities: pd.DataFrame, success_tag: str = "success"):
        """
        Parameters
        ----------
        probabilities
            A Pandas DataFrame containing two columns 'bin_start' and
            'bin_end', and an arbitrary number of additional columns.
            'bin_start' and 'bin_end' correspond to the left and right limits
            respectively of left-inclusive, right-exclusive bins of relative
            time position (e.g. if total duration is 10 seconds, 3 seconds
            would fall in bin [0.0, 0.5) and 7 seconds in bin [0.5, 1.0)).
            All other columns are interpreted as relative probabilities for a
            tag (identified by the column name) within a bin.
            All probabilities for tags in a bin MUST add up to 1.0.

            For example, a row<br><br>

            <table>
            <thead>
              <tr>
                <th>bin_start</th>
                <th>bin_end</th>
                <th>repeat</th>
                <th>low_confidence</th>
                <th>blank</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>0.0</td>
                <td>0.2</td>
                <td>0.3</td>
                <td>0.1</td>
                <td>0.6</td>
              </tr>
            </tbody>
            </table><br>

            indicates that within bin [0, 0.2), 'repeat' frames occur with a
            relative probability of 0.3, 'low_confidence' frames with a
            relative probability of 0.1, and 'blank' frames with a relative
            probability of 0.6.
        success_tag
            String to be returned by methods of this class whenever the target
            step time has been achieved.

        """

        # validation

        columns = set(probabilities.columns)

        try:
            columns.remove("bin_start")
            columns.remove("bin_end")
        except KeyError:
            raise RuntimeError(
                "Probability dataframe must include bin_start and bin_end columns."
            )

        prob_sums = np.zeros(len(probabilities.index))

        for column in columns:
            prob_sums += probabilities[column]

        if not np.all(np.isclose(prob_sums, 1.0)):
            raise RuntimeError(
                "Sum of probabilities for each bin must be equal to 1.0."
            )

        # process probabilities
        self._probs = probabilities.copy()
        self._probs["interval"] = pd.IntervalIndex.from_arrays(
            left=probabilities["bin_start"],
            right=probabilities["bin_end"],
            closed="left",
        )
        self._probs = self._probs.drop(columns=["bin_start", "bin_end"]).set_index(
            "interval", verify_integrity=True
        )

        self._rng = np.random.default_rng()
        self._success_tag = success_tag

    def _sample_from_distribution(self, rel_pos: float) -> str:
        if rel_pos > 1:
            return self._success_tag

        probs = self._probs[self._probs.index.contains(rel_pos)].iloc[0]
        return self._rng.choice(a=probs.index, replace=False, p=probs.values)

    def get_frame_at_instant(self, instant: float | int, step_time: float | int) -> str:
        """
        Return a frame sampled from a specific instant in a step.

        Parameters
        ----------
        instant
            Number of seconds since the start of the step.
        step_time
            Total target step duration.

        Returns
        -------
        str
            A randomly sampled step tag.
        """

        # purely according to distributions
        try:
            return self._sample_from_distribution(float(instant) / float(step_time))
        except ZeroDivisionError:
            # if step time is 0 we can immediately assume step is over!
            return self._success_tag

    @abc.abstractmethod
    def step_iterator(
        self,
        target_time: float,
        delay: float,
        infinite: bool = False,
    ) -> Iterator[Tuple[str, float]]:
        pass


class ZeroWaitFrameSamplingModel(BaseFrameSamplingModel):
    def step_iterator(
        self,
        target_time: float,
        delay: float,
        infinite: bool = False,
    ) -> Iterator[Tuple[str, float]]:
        """
        An iterator over the frame tags in a step.
        Any calls to next() between instants 0 and target_time will
        correspond to frame tags sampled from the internal distributions.
        Calls to next() after a time greater than target time has been
        elapsed will always return a success tag; if infinite is False, the iterator
        will additionally be closed.

        Yields
        ------
        str
            Frame tags.
        """

        step_start = time.monotonic()
        while True:
            instant = time.monotonic() - step_start
            yield self.get_frame_at_instant(instant, target_time), instant
            if instant > target_time and not infinite:
                return


class IdealFrameSamplingModel(ZeroWaitFrameSamplingModel):
    def step_iterator(
        self,
        target_time: float,
        delay: float,
        infinite: bool = False,
    ) -> Iterator[Tuple[str, float]]:
        step_start = time.monotonic()
        while True:
            time.sleep(max(target_time - (time.monotonic() - step_start), 0))
            dt = time.monotonic() - step_start
            yield self.get_frame_at_instant(dt, target_time), dt
            if dt > target_time and not infinite:
                return


class HoldFrameSamplingModel(ZeroWaitFrameSamplingModel):
    """
    Doesn't sample for a specified period of time at the beginning of each step.
    """

    def __init__(
        self,
        probabilities: pd.DataFrame,
        hold_time_seconds: float,
        success_tag: str = "success",
    ):
        super(HoldFrameSamplingModel, self).__init__(
            probabilities, success_tag=success_tag
        )
        self._hold_time = hold_time_seconds

    def step_iterator(
        self,
        target_time: float,
        delay: float,
        infinite: bool = False,
    ) -> Iterator[Tuple[str, float]]:
        step_start = time.monotonic()
        time.sleep(self._hold_time)
        while True:
            instant = time.monotonic() - step_start
            yield self.get_frame_at_instant(instant, target_time), instant
            if instant > target_time and not infinite:
                return


class RegularFrameSamplingModel(ZeroWaitFrameSamplingModel):
    """
    Samples in constant discrete time intervals. Defaults to zero-wait sampling if
    the time between calls to the step iterator is longer than the sampling interval!
    """

    def __init__(
        self,
        probabilities: pd.DataFrame,
        sampling_interval_seconds: float,
        success_tag: str = "success",
    ):
        super(RegularFrameSamplingModel, self).__init__(
            probabilities, success_tag=success_tag
        )
        self._interval = sampling_interval_seconds

    def step_iterator(
        self,
        target_time: float,
        delay: float,
        infinite: bool = False,
    ) -> Iterator[Tuple[str, float]]:
        step_start = time.monotonic()
        time.sleep(self._interval)

        while True:
            t_sample = time.monotonic()
            instant = t_sample - step_start
            yield self.get_frame_at_instant(instant, target_time), instant
            if instant > target_time and not infinite:
                return

            dt = time.monotonic() - t_sample
            time.sleep(max(0.0, self._interval - dt))
