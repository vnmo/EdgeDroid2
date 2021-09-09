from collections import deque
from unittest import TestCase

import numpy as np
import pandas as pd

from edgedroid.execution_times import Binner, _calculate_impairment_chunks, \
    preprocess_data


class TestDataPreprocessing(TestCase):
    impairment_test = {
        'impairment'     : pd.Series([
            1, 1, 2, 2, 2, 0, 0, 3, 0, 0, 1, 1
        ], dtype=pd.CategoricalDtype(categories=[0, 1, 2, 3], ordered=True)),
        'prev_transition': np.array([
            0, 0, 0, 1, 1, 1, -1, -1, 1, -1, -1, 1
        ]),
        'prev_duration'  : np.array([
            np.nan, 1, 2, 1, 2, 3, 1, 2, 1, 1, 2, 1
        ])
    }

    # test_data = pd.DataFrame({
    #     'run_id'     : [1, 1, 2, 2, 2, 3, 3],
    #     'neuroticism': [1.5, 1.5, 0.5, 0.5, 0.5, 2.5, 2.5],
    #     'step_seq'   : [1, 2, 1, 2, 3, 1, 2],
    #     'exec_time'  : [1, 1, 1, 2, 3, 1, 2],
    #     'delay'      : [0.5, 1.5, 2.5, 1.5, 2.5, 0.5, 0.5],
    # })

    test_data = pd.DataFrame({
        'run_id'     : [1, 1, 1, 1, 1, 1],
        'neuroticism': [0.5, 1.5, 2.5, 0.5, 1.5, 2.5],
        'step_seq'   : [1, 2, 3, 4, 5, 6],
        'exec_time'  : [1, 1, 2, 2, 3, 3],
        'delay'      : [1.5, 0.5, 0.5, 0.5, 2.5, 2.5],
    })

    exp_data = pd.DataFrame({
        'run_id'         : [1, 1, 1, 1, 1, 1],
        'step_seq'       : [1, 2, 3, 4, 5, 6],
        'exec_time'      : [1, 1, 2, 2, 3, 3],
        'neuroticism'    : [0, 1, 2, 0, 1, 2],
        'prev_impairment': [-1, 1, 0, 0, 0, 2],
        'prev_duration'  : [-1, 0, 0, 0, 1, 0],
        'transition'     : [0, 0, -1, -1, -1, 1]
    })

    neuro_bins = np.array([-np.inf, 1, 2, np.inf])
    delay_bins = np.array([-np.inf, 1, 2, np.inf])
    duration_bins = np.array([-np.inf, 2, np.inf])

    def test_impairment_chunks(self) -> None:
        prev_trans, prev_dur = _calculate_impairment_chunks(
            impairment=self.impairment_test['impairment'])

        np.testing.assert_array_equal(
            self.impairment_test['prev_transition'],
            prev_trans
        )
        np.testing.assert_array_equal(
            self.impairment_test['prev_duration'],
            prev_dur
        )

    def test_preprocess_data(self) -> None:
        test_data = self.test_data.set_index(['run_id', 'step_seq'],
                                             verify_integrity=True)
        exp_data = self.exp_data.set_index(['run_id', 'step_seq'],
                                           verify_integrity=True)

        result = preprocess_data(
            neuroticism_bins=self.neuro_bins,
            impairment_bins=self.delay_bins,
            duration_bins=self.duration_bins,
            execution_time_data=test_data
        )

        pd.testing.assert_frame_equal(result.data, exp_data,
                                      check_dtype=False)

        # very simple, multi-run test
        test_data = deque()
        exp_data = deque()
        for i in range(40):
            test_df = self.test_data.copy()
            test_df['run_id'] = i
            test_data.append(test_df)

            exp_df = self.exp_data.copy()
            exp_df['run_id'] = i
            exp_data.append(exp_df)

        test_data = pd.concat(test_data, ignore_index=True) \
            .set_index(['run_id', 'step_seq'], verify_integrity=True)
        exp_data = pd.concat(exp_data, ignore_index=True) \
            .set_index(['run_id', 'step_seq'], verify_integrity=True)

        result = preprocess_data(
            neuroticism_bins=self.neuro_bins,
            impairment_bins=self.delay_bins,
            duration_bins=self.duration_bins,
            execution_time_data=test_data
        )

        pd.testing.assert_frame_equal(result.data, exp_data,
                                      check_dtype=False)

    def test_binner(self) -> None:
        # test the Binner class
        # two bins: [1, 2) and [2, 3)
        bins = [1, 2, 3]

        binner = Binner(bin_edges=bins)

        with self.assertRaises(Binner.BinningError):
            # falls outside of left edge
            binner.bin(0)

        with self.assertRaises(Binner.BinningError):
            # falls outside of right edge
            binner.bin(4)

        with self.assertRaises(Binner.BinningError):
            # ranges are right-exclusive
            binner.bin(3)

        self.assertEqual(binner.bin(1.5), 0)
        self.assertEqual(binner.bin(2.7), 1)
        self.assertEqual(binner.bin(1), 0)
        self.assertEqual(binner.bin(2), 1)


