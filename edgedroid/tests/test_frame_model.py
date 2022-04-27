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

import numpy as np
import pandas as pd
from numpy import testing as nptesting

from ..models import FrameModel
from ..data import load_default_frame_probabilities


class TestFrameModel(unittest.TestCase):
    def setUp(self) -> None:
        # load data
        self.probs = load_default_frame_probabilities()

        self.model = FrameModel(self.probs)

        self.probs["interval"] = pd.IntervalIndex.from_arrays(
            left=self.probs["bin_start"],
            right=self.probs["bin_end"],
            closed="left",
        )
        self.probs = self.probs.drop(columns=["bin_start", "bin_end"]).set_index(
            "interval", verify_integrity=True
        )
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
                rel_pos = self.rng.uniform(
                    rel_pos_bin.left, min(rel_pos_bin.right, 1.0)
                )
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
