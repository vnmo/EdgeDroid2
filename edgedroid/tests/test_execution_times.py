import unittest
from typing import Any, Dict, Sequence
from unittest import TestCase

import nptyping
import numpy as np
import pandas as pd
from numpy import testing as nptest
from pandas import arrays
from tqdm import tqdm

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
                "delay": [1, 2, 3],
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
                "delay": delays,
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
        transitions = [
            Transition.NONE.value,
            Transition.NONE.value,
            Transition.L2H.value,
            Transition.L2H.value,
            Transition.H2L.value,
            Transition.L2H.value,
            Transition.H2L.value,
            Transition.L2H.value,
        ]

        imp_bins = arrays.IntervalArray.from_breaks(
            [0, 1, 2, 3, np.inf], closed="right"
        )
        bins = arrays.IntervalArray.from_breaks([-np.inf, np.inf], closed="both")

        data = pd.DataFrame(
            {
                "seq": np.arange(len(delays)) + 1,
                "run_id": [0] * len(delays),
                "delay": delays,
                "neuroticism": [0] * len(delays),
                "exec_time": [0] * len(delays),
            }
        )

        proc_data = preprocess_data(data, bins, imp_bins, bins)
        nptest.assert_array_equal(proc_data.transition, transitions)

    def test_fade_distance(self):
        # TODO
        pass


class TestModels(unittest.TestCase):
    def setUp(self) -> None:
        self.fade_distance = 4
        raw_data_params = load_default_exec_time_data()
        self.data = preprocess_data(
            *raw_data_params, transition_fade_distance=self.fade_distance
        )
        self.raw_data, *_ = raw_data_params

    def _test_model_states(
        self,
        model: ExecutionTimeModel,
        delays: nptyping.NDArray,
        expected_states: Sequence[Dict[str, Any]],
    ):
        model.reset()
        self.assertEqual(len(delays), len(expected_states))

        # check initial state
        model_state = model.state_info()

        # see https://stackoverflow.com/a/59777678
        self.assertEqual(expected_states[0], model_state)
        prev_state = model_state

        # iterate over the rest of the steps and match the states
        for delay, state in zip(delays[:-1], expected_states[1:]):
            model.set_delay(delay)
            model_state = model.state_info()
            self.assertEqual(state, model_state, prev_state)
            prev_state = model_state

    def test_states_step_by_step(self):
        for mcls in tqdm((EmpiricalExecutionTimeModel, TheoreticalExecutionTimeModel)):
            for run_id, df in tqdm(self.raw_data.groupby("run_id")):
                neuro = df.iloc[0]["neuroticism"]

                model = mcls(
                    self.data,
                    neuroticism=neuro,
                    transition_fade_distance=self.fade_distance,
                )

                delays = df.delay.to_numpy()
                states = (
                    self.data[self.data.run_id == run_id]
                    .drop(columns=["next_exec_time", "run_id"])
                    .to_dict("records")
                )

                self._test_model_states(model, delays, states)
