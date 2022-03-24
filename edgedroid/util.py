from typing import Sequence

import numpy.typing as npt
import numpy as np

from .execution_times import ExecutionTimeModel


def gen_model_trace(
    delays: Sequence[int | float], model: ExecutionTimeModel
) -> npt.NDArray[int | float]:
    exec_times = np.empty(len(delays) + 1)

    # initial time
    exec_times[0] = model.get_execution_time()

    for i, d in enumerate(delays):
        model.set_delay(d)
        exec_times[i + 1] = model.get_execution_time()

    return exec_times
