import abc
from typing import Optional

import numpy as np
import pandas as pd

from . import data as edgedroid_data


class ExecutionTimeModel(abc.ABC):
    def __init__(self,
                 data: pd.DataFrame,
                 neuro_bins: np.ndarray,
                 impair_bins: np.ndarray,
                 duration_bins: np.ndarray):
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
                 neuroticism_bins: Optional[np.ndarray] = None,
                 impairment_bins: Optional[np.ndarray] = None,
                 duration_bins: Optional[np.ndarray] = None,
                 data: Optional[pd.DataFrame] = None):
        self.data = data if data is not None else \
            edgedroid_data.load_default_exec_time_data()

        pass

    def make_model(self,
                   neuroticism: float,
                   empirical: bool = False) -> ExecutionTimeModel:
        pass
