import unittest

import numpy as np
import pandas as pd
from numpy import testing as nptesting

from edgedroid.frames import FrameModel


class TestFrameModel(unittest.TestCase):
    def setUp(self) -> None:
        # load data
        self.probs = pd.read_csv(
            '../edgedroid/data/resources/frame_probabilitites.csv')
        self.model = FrameModel(self.probs)

        self.probs['interval'] = pd.IntervalIndex.from_arrays(
            left=self.probs['bin_start'],
            right=self.probs['bin_end'],
            closed='left',
        )
        self.probs = self.probs \
            .drop(columns=['bin_start', 'bin_end']) \
            .set_index('interval', verify_integrity=True)
        self.rng = np.random.default_rng()

        self.num_samples = 2000
        self.comp_thresh = 0.05

    def test_deterministic_model(self) -> None:
        # for each bin, generate a large amount of samples and then compare
        # with the underlying data

        for rel_pos_bin, bin_probs in self.probs.iterrows():
            results = pd.Series(dtype=np.float64)

            for _ in range(self.num_samples):
                # get a random relative position within the bin
                rel_pos = self.rng.uniform(rel_pos_bin.left,
                                           min(rel_pos_bin.right, 1.0))
                # we min() the right side and 1.0 because we don't want any
                # success frames in this test

                frame = self.model.get_frame_at_instant(
                    instant=rel_pos,
                    step_time=1.0,
                )

                results[frame] = results.get(frame, 0) + 1

            results = results / results.sum()
            diff = (results - bin_probs).abs()
            nptesting.assert_array_less(diff, self.comp_thresh)
