from unittest import TestCase

import numpy as np
import pandas as pd
from numpy import testing as nptest
from pandas import arrays

from edgedroid.execution_times import ModelException, Transition, preprocess_data


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
                "run_id": [0] * len(delays),
                "delay": delays,
                "neuroticism": [0] * len(delays),
                "exec_time": [0] * len(delays),
            }
        )

        result = preprocess_data(data, bins, bins, dur_bins)

        for i, dur in enumerate(result.duration):
            expected = dur_bins[dur_bins.contains(i + 1)][0]
            self.assertEqual(dur, expected)

    def test_transitions(self):
        # test for transitions

        delays = [1, 2, 3, 1, 2, 1, 3, 2]
        transitions = [
            Transition.NONE.value,
            Transition.L2H.value,
            Transition.L2H.value,
            Transition.H2L.value,
            Transition.L2H.value,
            Transition.H2L.value,
            Transition.L2H.value,
            Transition.H2L.value,
        ]

        imp_bins = arrays.IntervalArray.from_breaks(
            [0, 1, 2, 3, np.inf], closed="right"
        )
        bins = arrays.IntervalArray.from_breaks([-np.inf, np.inf], closed="both")

        data = pd.DataFrame(
            {
                "run_id": [0] * len(delays),
                "delay": delays,
                "neuroticism": [0] * len(delays),
                "exec_time": [0] * len(delays),
            }
        )

        proc_data = preprocess_data(data, bins, imp_bins, bins)
        nptest.assert_array_equal(proc_data.transition, transitions)
