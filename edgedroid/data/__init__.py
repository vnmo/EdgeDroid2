from importlib import resources
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import arrays

_default_neuro_bins = np.array([-np.inf, 1 / 3, 2 / 3, np.inf])
_default_impairment_bins = np.array([-np.inf, 1.0, 2.0, np.inf])
_default_duration_bins = np.array([0, 5, 10, np.inf])


def load_default_exec_time_data() -> \
        Tuple[pd.DataFrame, arrays.IntervalArray,
              arrays.IntervalArray, arrays.IntervalArray]:
    from . import exec_times
    data_file = resources \
        .files(exec_times) \
        .joinpath('model_exec_times.parquet')

    return pd.read_parquet(data_file), \
           arrays.IntervalArray.from_breaks(_default_neuro_bins,
                                            closed='left'), \
           arrays.IntervalArray.from_breaks(_default_impairment_bins,
                                            closed='left'), \
           arrays.IntervalArray.from_breaks(_default_duration_bins,
                                            closed='left')


__all__ = ['load_default_exec_time_data']
