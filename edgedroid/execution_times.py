import abc
from typing import Optional

import numpy as np
import pandas as pd

from . import data as e_data


def preprocess_data(neuroticism_bins: np.ndarray = e_data.default_neuro_bins,
                    impairment_bins: np.ndarray =
                    e_data.default_impairment_bins,
                    duration_bins: np.ndarray = e_data.default_duration_bins,
                    execution_time_data: Optional[pd.DataFrame] = None) \
        -> pd.DataFrame:
    data = execution_time_data if execution_time_data is not None \
        else e_data.load_default_exec_time_data()

    # prepare the data
    # data['impairment'] = None
    data['prev_impairment'] = None
    data['prev_duration'] = 0
    data['transition'] = np.nan

    data['neuroticism'] = pd.cut(data['neuroticism'], neuroticism_bins)
    # data = data.reset_index()

    for _, df in data.groupby('run_id'):
        impairment = pd.cut(df['delay'], impairment_bins)
        df['transition'] = impairment \
            .cat.codes.diff(1) \
            .replace(np.nan, 0) \
            .astype(int)

        df['chunk'] = df['transition'].abs().ne(0).cumsum()
        for _, chunk in df.groupby('chunk'):
            df.loc[chunk.index, 'duration'] = \
                np.arange(len(chunk.index)) + 1

            transition = chunk['transition'].iloc[0]
            if transition != 0:
                df.loc[chunk.index, 'transition'] = transition
            else:
                df.loc[chunk.index, 'transition'] = np.nan

        data.loc[df.index, 'prev_impairment'] = impairment.shift()
        data.loc[df.index, 'prev_duration'] = df['duration'].shift()
        data.loc[df.index, 'transition'] = df['transition'].shift()

    data['transition'] = pd.Categorical(np.sign(data['transition']),
                                        ordered=True)
        # .rename_categories(['h2l', 'l2h'])

    data['prev_duration'] = pd.cut(data['prev_duration'],
                                   bins=duration_bins)
    data['prev_impairment'] = data['prev_impairment']\
        .astype('category').cat.codes
    data['prev_duration'] = data['prev_duration']\
        .astype('category').cat.codes

    return data


class ExecutionTimeModel(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_execution_time(self, delay: float) -> float:
        pass


class _EmpiricalExecutionTimeModel(ExecutionTimeModel):
    pass


class _TheoreticalExecutionTimeModel(ExecutionTimeModel):
    pass


class ExecutionTimeModelFactory:
    def __init__(self,
                 neuroticism_bins: np.ndarray = e_data.default_neuro_bins,
                 impairment_bins: np.ndarray = e_data.default_impairment_bins,
                 duration_bins: np.ndarray = e_data.default_duration_bins,
                 execution_time_data: Optional[pd.DataFrame] = None):
        pass

    def make_model(self,
                   neuroticism: float,
                   empirical: bool = False) -> ExecutionTimeModel:
        pass
