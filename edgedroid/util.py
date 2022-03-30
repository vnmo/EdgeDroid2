from collections import deque
from typing import Sequence

import numpy as np
import pandas as pd

from .execution_times import ExecutionTimeModel


def gen_model_trace(
    delays: Sequence[int | float], model: ExecutionTimeModel
) -> pd.DataFrame:
    """
    TODO: Document

    Parameters
    ----------
    delays
    model

    Returns
    -------

    """
    # exec_times = np.empty(len(delays) + 1)

    df_rows = deque()

    # initial time
    state = model.state_info()
    state["exec_time"] = model.get_execution_time()
    state["delay"] = np.nan
    df_rows.append(state)

    # exec_times[0] = model.get_execution_time()

    for i, d in enumerate(delays):
        model.set_delay(d)
        # exec_times[i + 1] = model.get_execution_time()
        state = model.state_info()
        state["exec_time"] = model.get_execution_time()
        state["delay"] = d
        df_rows.append(state)

    return pd.DataFrame(df_rows)
