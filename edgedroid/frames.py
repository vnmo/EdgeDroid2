from __future__ import annotations

import time
from collections import deque
from os import PathLike
from typing import Any, Dict, Iterator, Sequence, Tuple

import nptyping as npt
import numpy as np
import pandas as pd
import yaml


class FrameSet:
    def __init__(self,
                 name: str,
                 initial_frame: npt.NDArray,
                 steps: Sequence[Dict[str, npt.NDArray]]):
        self._name = name
        self._init_frame = initial_frame
        self._steps = tuple(steps)
        self._num_steps = len(self._steps)

    def __str__(self) -> str:
        return yaml.safe_dump(
            {
                'name'         : self._name,
                'num_steps'    : self.step_count,
                'initial_frame': f'{len(self._init_frame.tobytes())} bytes',
                'steps'        : [
                    {
                        tag: f'{len(data.tobytes())} bytes'
                        for tag, data in step.items()
                    } for step in self._steps
                ]
            }
        )

    @property
    def step_count(self) -> int:
        return self._num_steps

    @property
    def name(self) -> str:
        return self._name

    def get_initial_frame(self) -> npt.NDArray:
        return self._init_frame.copy()

    def get_frame(self,
                  step_index: Any,
                  frame_tag: str) -> npt.NDArray:
        return self._steps[step_index][frame_tag].copy()

    @classmethod
    def from_datafile(cls,
                      task_name: str,
                      trace_path: PathLike | str) -> FrameSet:
        """
        Opens a frame datafile and parses it.

        TODO: Add specification on frame datafiles.

        Parameters
        ----------
        task_name
            Task name for this trace.
        trace_path
            Path to the datafile.

        Returns
        -------

        """

        data = np.load(trace_path)

        # trace NPZ file contains initial frame + 3 frames per step
        # success, blank, and low_confidence
        assert (len(data) - 1) % 3 == 0
        num_steps = (len(data) - 1) // 3

        init_frame = data['initial']
        # TODO: hardcoded categories
        steps = deque()
        for step in range(num_steps):
            step_dict = {}
            for tag in ('success', 'blank', 'low_confidence'):
                step_dict[tag] = data[f'step{step:02d}_{tag}']
            steps.append(step_dict)

        return FrameSet(
            name=task_name,
            initial_frame=init_frame,
            steps=steps
        )


class FrameModel:
    def __init__(self,
                 probabilities: pd.DataFrame,
                 success_tag: str = 'success'):
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
            columns.remove('bin_start')
            columns.remove('bin_end')
        except KeyError:
            raise RuntimeError('Probability dataframe must include bin_start '
                               'and bin_end columns.')

        prob_sums = np.zeros(len(probabilities.index))

        for column in columns:
            prob_sums += probabilities[column]

        if not np.all(np.isclose(prob_sums, 1.0)):
            raise RuntimeError('Sum of probabilities for each bin must be '
                               'equal to 1.0.')

        # process probabilities
        self._probs = probabilities.copy()
        self._probs['interval'] = pd.IntervalIndex.from_arrays(
            left=probabilities['bin_start'],
            right=probabilities['bin_end'],
            closed='left',
        )
        self._probs = self._probs \
            .drop(columns=['bin_start', 'bin_end']) \
            .set_index('interval', verify_integrity=True)

        self._rng = np.random.default_rng()
        self._success_tag = success_tag

    def _sample_from_distribution(self, rel_pos: float) -> str:
        if rel_pos > 1:
            return self._success_tag

        probs = self._probs[self._probs.index.contains(rel_pos)].iloc[0]
        return self._rng.choice(
            a=probs.index,
            replace=False,
            p=probs.values
        )

    def get_frame_at_instant(self,
                             instant: float,
                             step_time: float) -> str:
        # purely according to distributions
        return self._sample_from_distribution(instant / step_time)

    def step_iterator(self,
                      target_time: float) -> Iterator[Tuple[str, float]]:
        """
        Returns
        -------
        """

        step_start = time.monotonic()
        while True:
            instant = time.monotonic() - step_start
            yield self.get_frame_at_instant(instant, target_time), instant
