from importlib import resources

import pandas as pd


def load_default_exec_time_data() -> pd.DataFrame:
    from . import exec_times
    data_file = resources \
        .files(exec_times) \
        .joinpath('model_exec_times.parquet')

    return pd.read_parquet(data_file)
