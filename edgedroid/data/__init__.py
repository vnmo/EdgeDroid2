from importlib import resources

import numpy as np
import pandas as pd


def load_default_exec_time_data() -> pd.DataFrame:
    from . import exec_times
    data_file = resources \
        .files(exec_times) \
        .joinpath('model_exec_times.parquet')

    return pd.read_parquet(data_file)


default_neuro_bins = np.array([-np.inf, 1 / 3, 2 / 3, np.inf])
default_impairment_bins = np.array([-np.inf, 1.0, 2.0, np.inf])
default_duration_bins = np.array([0, 5, 13, np.inf])

__all__ = ['load_default_exec_time_data',
           default_duration_bins,
           default_impairment_bins,
           default_neuro_bins]
