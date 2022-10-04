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

import unittest
from unittest import TestCase

import nptyping
import numpy as np
import pandas as pd
from numpy import testing as nptest
from pandas import arrays
from tqdm import tqdm
from typing import Any, Dict, Sequence

from ..data import *
from ..models.timings import (
    EmpiricalExecutionTimeModel,
    ExecutionTimeModel,
    ModelException,
    TheoreticalExecutionTimeModel,
    Transition,
    preprocess_data,
)


class TestDataPreprocessing(TestCase):
    def test_missing_col(self):
        # base test for missing columns
        input_df = pd.DataFrame(
            {
                "run_id": [1, 2, 3],
                "neuroticism": [1, 2, 3],
                "exec_time": [1, 2, 3],
                "ttf": [1, 2, 3],
            }
        )

        for col in input_df.columns:
            df = input_df.copy().drop(columns=[col])
            with self.assertRaises(ModelException):
                preprocess_data(
                    exec_time_data=df,
                    neuro_bins=arrays.IntervalArray.from_breaks([-np.inf, np.inf]),
                    impair_bins=arrays.IntervalArray.from_breaks([-np.inf, np.inf]),
                    duration_bins=arrays.IntervalArray.from_breaks([-np.inf, np.inf]),
                )

    def test_track_duration(self):
        # test for duration tracking and binning

        delays = [1] * 10
        bins = arrays.IntervalArray.from_breaks([-np.inf, np.inf], closed="both")
        dur_bins = arrays.IntervalArray.from_breaks([0, 3, 6, 9, np.inf], closed="left")

        data = pd.DataFrame(
            {
                "seq": np.arange(len(delays)) + 1,
                "run_id": [0] * len(delays),
                "ttf": delays,
                "neuroticism": [0] * len(delays),
                "exec_time": [0] * len(delays),
            }
        )

        result = preprocess_data(data, bins, bins, dur_bins)

        for i, dur in enumerate(result.duration):
            expected = dur_bins[dur_bins.contains(i + 1)][0]
            self.assertEqual(dur, expected, result)

    def test_transitions(self):
        # test for transitions

        delays = [1, 2, 3, 1, 2, 1, 3, 2]
        transitions = np.array(
            [
                Transition.NONE.value,
                Transition.NONE.value,
                Transition.L2H.value,
                Transition.L2H.value,
                Transition.H2L.value,
                Transition.L2H.value,
                Transition.H2L.value,
                Transition.L2H.value,
            ],
            dtype=object,
        )

        imp_bins = arrays.IntervalArray.from_breaks(
            [0, 1, 2, 3, np.inf], closed="right"
        )
        bins = arrays.IntervalArray.from_breaks([-np.inf, np.inf], closed="both")

        data = pd.DataFrame(
            {
                "seq": np.arange(len(delays)) + 1,
                "run_id": [0] * len(delays),
                "ttf": delays,
                "neuroticism": [0] * len(delays),
                "exec_time": [0] * len(delays),
            }
        )

        proc_data = preprocess_data(data, bins, imp_bins, bins)
        nptest.assert_array_equal(proc_data.transition, transitions)

    def test_fade_distance(self):
        # TODO
        pass

    def test_truncating(self):
        task_name = "square00"
        good_truncs = (1, 5, 10, 30, 150)
        bad_truncs = (180, 200)

        self.assertEqual(
            load_default_trace(task_name).step_count,
            load_default_trace(task_name, truncate=-100).step_count,
        )
        self.assertEqual(
            len(load_default_task(task_name)),
            len(load_default_task(task_name, truncate=-100)),
        )

        for trunc in good_truncs:
            trace = load_default_trace(task_name, truncate=trunc)
            self.assertEqual(trunc, trace.step_count)  # plus initial trigger step

            task = load_default_task(task_name, truncate=trunc)
            self.assertEqual(trunc, len(task))

        for trunc in bad_truncs:
            with self.assertRaises(Exception):
                load_default_trace(task_name, truncate=trunc)

            with self.assertRaises(Exception):
                load_default_task(task_name, truncate=trunc)


class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        # self.fade_distance = 8
        raw_data_params = load_default_exec_time_data()
        self.data = preprocess_data(
            *raw_data_params,
            # transition_fade_distance=self.fade_distance,
        )
        self.raw_data, *_ = raw_data_params

    def _test_model_states(
        self,
        model: ExecutionTimeModel,
        ttfs: nptyping.NDArray,
        expected_states: Sequence[Dict[str, Any]],
    ):
        model.reset()
        self.assertEqual(len(ttfs), len(expected_states))

        # check initial state
        # this should fail
        with self.assertRaises(ModelException):
            model_state = model.state_info()

        # see https://stackoverflow.com/a/59777678
        # self.assertEqual(expected_states[0], model_state)
        prev_state = None

        # iterate over the rest of the steps and match the states
        for ttf, state in zip(ttfs, expected_states):
            model.advance(ttf)
            model_state = model.state_info()
            self.assertEqual(state, model_state, prev_state)
            prev_state = model_state

    def test_states_step_by_step(self):
        for run_id, raw_df in tqdm(self.raw_data.groupby("run_id")):
            neuro = raw_df.iloc[0]["neuroticism"]

            model = EmpiricalExecutionTimeModel(
                self.data,
                neuroticism=neuro,
                state_checks_enabled=False,
                # transition_fade_distance=self.fade_distance,
            )

            ttfs = raw_df.ttf.shift().fillna(0).to_numpy()
            states = (
                self.data[self.data.run_id == run_id]
                .drop(columns=["next_exec_time", "run_id"])
                .copy()
            )

            states["seq"] = np.arange(1, len(states.index) + 1)
            self._test_model_states(model, ttfs, states.to_dict("records"))

    def test_no_neuroticism(self):
        rng = np.random.default_rng()

        # just check if setting neuroticism to None causes problems
        for model_cls in (EmpiricalExecutionTimeModel, TheoreticalExecutionTimeModel):
            model = model_cls.from_default_data(neuroticism=None)

            for _ in range(100):
                model.advance((rng.random() * (10.0 - 0.5)) + 0.5).get_execution_time()
                model.get_model_params()
                model.state_info()
                model.get_expected_execution_time()
