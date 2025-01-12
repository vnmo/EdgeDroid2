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
import time
import unittest
from typing import Optional

from loguru import logger
from tqdm import tqdm

from .. import data as e_data
from ..models import EdgeDroidModel, FrameTimings, ZeroWaitFrameSamplingModel
from ..models.timings import ConstantExecutionTimeModel
from gabriel_lego.api import FrameResult, LEGOTask


class EndToEndTest(unittest.TestCase):
    def test_edgedroid_model(self) -> None:
        # simple end to end test with a constant execution time model
        # intended to check that the EdgeDroid model is generating the correct
        # success frames without having to wait for 20 minutes
        # also checks truncating

        trace = f"square00"
        for tlen in (5, 50, -1):
            logger.debug(f"Testing trace {trace} (truncated to {tlen})")
            frameset = e_data.load_default_trace(trace, truncate=tlen)
            frame_model = ZeroWaitFrameSamplingModel(
                e_data.load_default_frame_probabilities()
            )

            timing_model = ConstantExecutionTimeModel(0)

            model = EdgeDroidModel(
                frame_trace=frameset, frame_model=frame_model, timing_model=timing_model
            )

            task = LEGOTask(e_data.load_default_task(trace, truncate=tlen))
            for model_step in tqdm(model.play_steps(), total=task.task_length):
                self.assertFalse(task.finished)

                frame_timings: Optional[FrameTimings] = None
                while True:
                    try:
                        model_frame = model_step.send(frame_timings)
                    except StopIteration:
                        break
                    ti = time.monotonic()
                    self.assertEqual(
                        FrameResult.SUCCESS,
                        task.submit_frame(model_frame.frame_data),
                        model_frame,
                    )
                    proctime = time.monotonic() - ti
                    frame_timings = FrameTimings(0, proctime)

            self.assertTrue(task.finished)
            logger.success(f"Trace {trace} passed test")
            logger.debug(f"Step metrics:\n{model.model_step_metrics()}")
