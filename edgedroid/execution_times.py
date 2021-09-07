from __future__ import annotations

import abc
from typing import Generator, Iterator, NamedTuple, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from . import data as e_data


# TODO: pydocs

class Binner:
    """
    Utility class for binning values into bins defined by an array of bin edges.
    Values will be binned into the bins defined by these such that `i` will
    be indicated as the bin for a `value` iff `value in [bin_edges[i],
    bin_edges[i + i])`

    Parameters
    ----------
    bin_edges
        A Sequence defining the bin edges.

    """
    class BinningError(Exception):
        def __init__(self, val: Union[float, int],
                     bin_edges: np.ndarray):
            b_edges = ', '.join([str(i) for i in bin_edges])
            super(Binner.BinningError, self).__init__(
                f'{val} does not fall within the defined bin edges [{b_edges})!'
            )

    def __init__(self, bin_edges: Sequence[Union[float, int]]):
        self._bin_edges = np.unique(bin_edges)

    def bin(self, value: Union[int, float]) -> int:
        """
        Bin a value into the bin edges stored in this binner.

        Values will be binned into the bin edges such that `i` will be
        indicated as the bin for a `value` iff `value in [bin_edges[i],
        bin_edges[i + i])`

        Parameters
        ----------
        value
            The value to bin.

        Returns
        -------
        int
            An index `i` such that `value in [bin_edges[i], bin_edges[i + i])`

        Raises
        ------
        Binner.BinningError
            If `value` is less than `bin_edges[0]` or greater than
            `bin_edges[-1]`.
        """
        bin_idx = int(np.digitize(value, self._bin_edges))
        if bin_idx <= 0 or bin_idx >= self._bin_edges.size:
            raise Binner.BinningError(value, self._bin_edges)
        return bin_idx - 1


class _PreprocessedData(NamedTuple):
    data: pd.DataFrame
    neuroticism_binner: Binner
    impairment_binner: Binner
    duration_binner: Binner


def preprocess_data(neuroticism_bins: np.ndarray = e_data.default_neuro_bins,
                    impairment_bins: np.ndarray =
                    e_data.default_impairment_bins,
                    duration_bins: np.ndarray = e_data.default_duration_bins,
                    execution_time_data: Optional[pd.DataFrame] = None) \
        -> _PreprocessedData:
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

    return _PreprocessedData(data,
                             Binner(bin_edges=neuroticism_bins),
                             Binner(bin_edges=impairment_bins),
                             Binner(bin_edges=duration_bins))


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

    @abc.abstractmethod
    def execution_time_iterator(self) -> _ExecTimeIterator:
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
    class _StepParameters(NamedTuple):
        impairment_lvl: int
        duration_lvl: int
        transition: int
        raw_duration: int

    def __init__(self,
                 data: pd.DataFrame,
                 neuro_level: int,
                 impair_binner: Binner,
                 dur_binner: Binner):
        super().__init__()
        # first, we filter on neuroticism
        data = data.loc[data.neuroticism == neuro_level]

        # next, prepare views
        self._data_views = {}

        # first, initial steps
        # these have a special key (None, None, None)
        init_data = data.loc[pd.IndexSlice[:, 1], :]
        self._data_views[(None, None, None)] = init_data
        data = data.loc[data.index.difference(init_data.index)]

        # next, other steps
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
        data = self._data_views[(None, None, None)]
        return data.exec_time.sample(1).values[0]

    def _calculate_parameters(self, delay: float) -> _StepParameters:
        try:
            impairment = self._impairment_binner.bin(delay)
        except Binner.BinningError as e:
            raise ModelException() from e

        if self._prev_impairment is None:
            # previous step was first step
            duration = 1
            transition = self._latest_transition
        elif impairment == self._prev_impairment:
            duration = self._duration + 1
            transition = self._latest_transition
        else:
            duration = 1
            transition = int(np.sign(impairment - self._prev_impairment))

        try:
            binned_duration = self._duration_binner.bin(duration)
        except Binner.BinningError as e:
            raise ModelException() from e

        return _EmpiricalExecutionTimeModel._StepParameters(impairment,
                                                            binned_duration,
                                                            transition,
                                                            duration)

    def get_execution_time(self, delay: float) -> float:
        params = self._calculate_parameters(delay)

        # get the appropriate data view
        data = self._data_views[(params.impairment_lvl,
                                 params.duration_lvl,
                                 params.transition)]

        # update state
        self._prev_impairment = params.impairment_lvl
        self._duration = params.raw_duration
        self._latest_transition = params.transition

        # finally, sample from the data and return an execution time in seconds
        return data.exec_time.sample(1).values[0]


class _TheoreticalExecutionTimeModel(_EmpiricalExecutionTimeModel):
    def __init__(self,
                 data: pd.DataFrame,
                 neuro_level: int,
                 impair_binner: Binner,
                 dur_binner: Binner):
        super(_TheoreticalExecutionTimeModel, self).__init__(data,
                                                             neuro_level,
                                                             impair_binner,
                                                             dur_binner)

        # at this point, the views have been populated with data according to
        # the binnings
        # now we fit distributions to each data view

        self._dists = {}
        for imp_dur_trans, df in self._data_views.items():
            # get the execution times, then fit an ExGaussian/ExponNorm
            # distribution to the samples

            exec_times = df['exec_time'].to_numpy()
            k, loc, scale = stats.exponnorm.fit(exec_times)

            self._dists[imp_dur_trans] = \
                stats.exponnorm.freeze(loc=loc, scale=scale, K=k)

    def get_initial_step_execution_time(self) -> float:
        # find initial distribution
        dist = self._dists[(None, None, None)]
        return dist.rvs()

    def get_execution_time(self, delay: float) -> float:
        params = self._calculate_parameters(delay)

        # get the appropriate distribution
        dist = self._dists[(params.impairment_lvl,
                            params.duration_lvl,
                            params.transition)]

        # update state
        self._prev_impairment = params.impairment_lvl
        self._duration = params.raw_duration
        self._latest_transition = params.transition

        # finally, sample from the dist and return an execution time in seconds
        return dist.rvs()


class ExecutionTimeModelFactory:
    def __init__(self,
                 neuroticism_bins: np.ndarray = e_data.default_neuro_bins,
                 impairment_bins: np.ndarray = e_data.default_impairment_bins,
                 duration_bins: np.ndarray = e_data.default_duration_bins,
                 execution_time_data: Optional[pd.DataFrame] = None):
        self._preprocessed_data = preprocess_data(
            neuroticism_bins=neuroticism_bins,
            impairment_bins=impairment_bins,
            duration_bins=duration_bins,
            execution_time_data=execution_time_data
        )

    def make_model(self,
                   neuroticism: float,
                   empirical: bool = False) -> ExecutionTimeModel:
        neuro_level = self._preprocessed_data \
            .neuroticism_binner.bin(neuroticism)

        if empirical:
            return _EmpiricalExecutionTimeModel(
                self._preprocessed_data.data,
                neuro_level,
                self._preprocessed_data.impairment_binner,
                self._preprocessed_data.duration_binner
            )
        else:
            return _TheoreticalExecutionTimeModel(
                self._preprocessed_data.data,
                neuro_level,
                self._preprocessed_data.impairment_binner,
                self._preprocessed_data.duration_binner
            )
