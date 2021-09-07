import abc
from typing import Generator, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import data as e_data


# TODO: pydocs

class Binner:
    class BinningError(Exception):
        def __init__(self, val: Union[float, int],
                     bin_edges: np.ndarray):
            b_edges = ', '.join([str(i) for i in bin_edges])
            super(Binner.BinningError, self).__init__(
                f'{val} does not fall within the defined bin edges [{b_edges})!'
            )

    def __init__(self, bin_edges: Sequence[Union[float, int]]):
        self._bin_edges = np.array(bin_edges)

    def bin(self, value: Union[int, float]) -> int:
        bin_idx = int(np.digitize(value, self._bin_edges))
        if bin_idx <= 0 or bin_idx >= self._bin_edges.size:
            raise Binner.BinningError(value, self._bin_edges)
        return bin_idx - 1


def preprocess_data(neuroticism_bins: np.ndarray = e_data.default_neuro_bins,
                    impairment_bins: np.ndarray =
                    e_data.default_impairment_bins,
                    duration_bins: np.ndarray = e_data.default_duration_bins,
                    execution_time_data: Optional[pd.DataFrame] = None) \
        -> Tuple[pd.DataFrame, Binner, Binner, Binner]:
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

    data['transition'] = np.sign(data['transition'])
    data['transition'] = data['transition'] \
        .fillna(0) \
        .astype('int') \
        .astype('category')
    # .rename_categories(['h2l', 'l2h'])

    data['prev_duration'] = pd.cut(data['prev_duration'],
                                   bins=duration_bins).cat.codes
    data['neuroticism'] = data['neuroticism'].cat.codes
    data['prev_impairment'] = data['prev_impairment'] \
        .astype('category').cat.codes

    return data, \
           Binner(bin_edges=neuroticism_bins), \
           Binner(bin_edges=impairment_bins), \
           Binner(bin_edges=duration_bins)


class ModelException(Exception):
    pass


class ExecutionTimeModel(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_initial_step_execution_time(self) -> float:
        pass

    @abc.abstractmethod
    def get_execution_time(self, delay: float) -> float:
        pass


class _ExecTimeIterator(Iterator[float]):
    def __init__(self, model: ExecutionTimeModel):
        self._delay = None

        def _generator() -> Generator[float]:
            # first call returns an execution time for an initial step
            yield model.get_initial_step_execution_time()

            # subsequent calls return values from non-initial steps
            while True:
                if self._delay is None:
                    raise ModelException('Must call set_delay() between '
                                         'iterations of the execution '
                                         'time generator!')

                yield model.get_execution_time(self._delay)
                self._delay = None

        self._gen = _generator()

    def set_delay(self, value: Union[int, float]) -> None:
        self._delay = float(value)

    def __next__(self) -> float:
        return next(self._gen)

    def next(self, delay: Optional[Union[int, float]]) -> float:
        self.set_delay(delay)
        return self.__next__()


class _EmpiricalExecutionTimeModel(ExecutionTimeModel):
    def __init__(self,
                 data: pd.DataFrame,
                 neuro_level: int,
                 impair_binner: Binner,
                 dur_binner: Binner):
        super().__init__()
        # first, we filter on neuroticism
        data = data.loc[data.neuroticism == neuro_level]

        # next, prepare views
        # first, initial steps
        self._init_data = data.loc[pd.IndexSlice[:, 1], :]
        data = data.loc[data.index.difference(self._init_data.index)]

        # next, other steps
        self._data_views = {}
        for imp_dur_trans, df in data.groupby(['prev_impairment',
                                               'prev_duration',
                                               'transition']):
            # imp_dur is a tuple (impairment, duration, transition)
            self._data_views[imp_dur_trans] = df

        self._impairment_binner = impair_binner
        self._duration_binner = dur_binner

        self._prev_impairment = None
        self._duration = 0
        self._latest_transition = 0

    def execution_time_iterator(self) -> Iterator[float]:
        return _ExecTimeIterator(model=self)

    def get_initial_step_execution_time(self) -> float:
        # sample from the data and return an execution time in seconds
        return self._init_data.exec_time.sample(1).values[0]

    def get_execution_time(self, delay: float) -> float:
        # start by binning delay
        try:
            impairment = self._impairment_binner.bin(delay)
        except Binner.BinningError as e:
            raise ModelException() from e

        if self._prev_impairment is None:
            # previous step was first step
            self._duration = 1
        elif impairment == self._prev_impairment:
            self._duration += 1
        else:
            self._duration = 1
            self._latest_transition = \
                int(np.sign(impairment - self._prev_impairment))

        try:
            binned_duration = self._duration_binner.bin(self._duration)
        except Binner.BinningError as e:
            raise ModelException() from e

        data = self._data_views[(impairment,
                                 binned_duration,
                                 self._latest_transition)]

        # update stored impairment
        self._prev_impairment = impairment

        # finally, sample from the data and return an execution time in seconds
        return data.exec_time.sample(1).values[0]


class _TheoreticalExecutionTimeModel(ExecutionTimeModel):
    pass


class ExecutionTimeModelFactory:
    def __init__(self,
                 neuroticism_bins: np.ndarray = e_data.default_neuro_bins,
                 impairment_bins: np.ndarray = e_data.default_impairment_bins,
                 duration_bins: np.ndarray = e_data.default_duration_bins,
                 execution_time_data: Optional[pd.DataFrame] = None):
        self._data = preprocess_data(
            neuroticism_bins=neuroticism_bins,
            impairment_bins=impairment_bins,
            duration_bins=duration_bins,
            execution_time_data=execution_time_data
        )

    def make_model(self,
                   neuroticism: float,
                   empirical: bool = False) -> ExecutionTimeModel:
        pass
