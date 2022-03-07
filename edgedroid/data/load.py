from importlib import resources
from importlib.resources import as_file
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import arrays

__all__ = [
    'load_default_frame_probabilities',
    'load_default_exec_time_data'
]

_default_neuro_bins = np.array([-np.inf, 1 / 3, 2 / 3, np.inf])
_default_impairment_bins = np.array([-np.inf, 1.0, 2.0, np.inf])
_default_duration_bins = np.array([0, 5, 10, np.inf])


def load_default_exec_time_data() -> \
        Tuple[pd.DataFrame, arrays.IntervalArray,
              arrays.IntervalArray, arrays.IntervalArray]:
    from . import resources as edgedroid_resources
    data_file = resources \
        .files(edgedroid_resources) \
        .joinpath('model_exec_times.parquet')

    with as_file(data_file) as fp:
        return pd.read_parquet(fp), \
               arrays.IntervalArray.from_breaks(_default_neuro_bins,
                                                closed='left'), \
               arrays.IntervalArray.from_breaks(_default_impairment_bins,
                                                closed='left'), \
               arrays.IntervalArray.from_breaks(_default_duration_bins,
                                                closed='left')


def load_default_frame_probabilities() -> pd.DataFrame:
    from . import resources as edgedroid_resources
    frame_prob_file = resources \
        .files(edgedroid_resources) \
        .joinpath('frame_probabilities.csv')
    with as_file(frame_prob_file) as fp:
        return pd.read_csv(fp)
