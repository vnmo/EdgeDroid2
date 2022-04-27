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

from loguru import logger

from .. import data as e_data
from ..models import EdgeDroidModel, FrameModel
from ..models.timings import ConstantExecutionTimeModel
from gabriel_lego.api import FrameResult, LEGOTask


class EndToEndTest(unittest.TestCase):
    def test_edgedroid_model(self) -> None:
        # simple end to end test with a constant execution time model
        # intended to check that the EdgeDroid model is generating the correct
        # success frames without having to wait for 20 minutes

        trace = f"test"
        logger.debug(f"Testing trace {trace}")
        frameset = e_data.load_default_trace(trace)
        frame_model = FrameModel(e_data.load_default_frame_probabilities())

        timing_model = ConstantExecutionTimeModel(0)

        model = EdgeDroidModel(
            frame_trace=frameset, frame_model=frame_model, timing_model=timing_model
        )

        task = LEGOTask(e_data.load_default_task(trace))
        for model_frame in model.play():
            self.assertEqual(
                FrameResult.SUCCESS,
                task.submit_frame(model_frame.frame_data),
                model_frame,
            )
            model.advance_step()
        logger.success(f"Trace {trace} passed test")
