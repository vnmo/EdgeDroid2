from __future__ import annotations

import abc
from typing import Any, Generator, Iterator, NamedTuple, Optional, Sequence, \
    Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from . import data as e_data

_NUM = Union[float, int]


# TODO: pydocs

class Binner:
    """
    Utility class for binning values into bins defined by an array of bin edges.
    Values will be binned into the bins defined by these such that `i` will
    be indicated as the bin for a `value` iff `value in (bin_edges[i],
    bin_edges[i + i]]`

    Parameters
    ----------
    bin_edges
        A Sequence defining the bin edges.

    """

    class BinningError(Exception):
        def __init__(self, val: npt.ArrayLike,
                     bin_edges: np.ndarray):
            super(Binner.BinningError, self).__init__(
                f'{val} do(es) not fall within the defined bin edges '
                f'{bin_edges}'
            )

    def __init__(self, bin_edges: Sequence[Union[float, int]]):
        self._bin_edges = np.unique(bin_edges)

    def bin(self, value: npt.ArrayLike) -> npt.ArrayLike:
        """
        Bin a value or a series of values into the bin edges stored in this
        binner.

        Values will be binned into the bin edges such that `i` will be
        indicated as the bin for a `value` iff `value in [bin_edges[i],
        bin_edges[i + i])`

        Parameters
        ----------
        value
            The value(s) to bin.

        Returns
        -------
        _ArrayLike
            An index `i` such that `value in [bin_edges[i], bin_edges[i + i])`

        Raises
        ------
        Binner.BinningError
            If `value` is less than `bin_edges[0]` or greater than
            `bin_edges[-1]`.
        """

        arr_value = np.atleast_1d(value)
        bin_indices = pd.cut(arr_value, self._bin_edges).codes
        if np.any(np.isnan(bin_indices)) \
                or np.any(bin_indices < 0) \
                or np.any(bin_indices >= self._bin_edges.size):
            raise Binner.BinningError(value, self._bin_edges)

        return bin_indices[0] if np.ndim(value) == 0 else bin_indices


class PreprocessedData(NamedTuple):
    data: pd.DataFrame
    neuroticism_binner: Binner
    impairment_binner: Binner
    duration_binner: Binner


def _process_impairment(impairment: npt.NDArray[Any]) \
        -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    transitions = np.empty(impairment.size + 1, dtype=int)
    durations = np.empty(impairment.size + 1, dtype=int)
    impairments = np.empty(impairment.size + 1, dtype=int)

    # special initial behaviors
    # transition before first step is 0
    # duration before first step is -1
    # impairment before first step is 0
    transitions[0] = 0
    durations[0] = -1
    impairments[0] = 0
    impairments[1:] = impairment

    for i in range(1, impairments.size):
        # compare with previous impairment and adjust transition and
        # duration accordingly
        transition = impairments[i] - impairments[i - 1]

        if transition == 0:
            # no change in impairment
            # increment duration by 1 (or set to 1 if prev duration was -1)
            duration = durations[i - 1] + 1 if durations[i - 1] > 0 else 1
            # also set transition to same transition as before
            transition = transitions[i - 1]
        else:
            # change in impairment
            # reset duration to 1, store transition
            duration = 1
            transition = int(np.sign(transition))

        # store values
        transitions[i] = transition
        durations[i] = duration
        impairments[i] = impairment[i - 1]

    return impairments[:-1], transitions[:-1], durations[:-1]


def preprocess_data(neuroticism_bins: np.ndarray = e_data.default_neuro_bins,
                    impairment_bins: np.ndarray =
                    e_data.default_impairment_bins,
                    duration_bins: np.ndarray = e_data.default_duration_bins,
                    execution_time_data: Optional[pd.DataFrame] = None) \
        -> PreprocessedData:
    """
    Preprocess a DataFrame with index (`run_id`, `step_seq`) and columns
    `exec_time`, `neuroticism`, and `delay` into a DataFrame appropriate for
    the model.

    Assumes the DataFrame rows are ORDERED.

    Any parameter not explicitly provided will be taken from the model defaults.

    Parameters
    ----------
    neuroticism_bins
        The bin edges for binning raw neuroticism values.
    impairment_bins
        The bin edges for binning raw delay values into levels.
    duration_bins
        The bin edges for binning duration values into levels.
    execution_time_data
        The DataFrame to be processed.

    Returns
    -------
    PreprocessedData
        The preprocessed data along with suitable Binner objects for values.
    """

    data = execution_time_data.copy() if execution_time_data is not None \
        else e_data.load_default_exec_time_data()

    # prepare the data
    # data['impairment'] = None
    data['prev_impairment'] = None
    data['prev_duration'] = 0
    data['transition'] = 0

    data['neuroticism'] = pd.cut(data['neuroticism'], neuroticism_bins)
    # data = data.reset_index()

    for _, df in data.groupby('run_id'):
        # grouping by subject, basically
        # bin delay
        impairment = pd.cut(df['delay'], impairment_bins).cat.codes.values
        prev_imp, prev_trans, prev_dur = _process_impairment(impairment)

        data.loc[df.index, 'prev_impairment'] = prev_imp
        data.loc[df.index, 'prev_duration'] = prev_dur
        data.loc[df.index, 'transition'] = prev_trans

    data['prev_duration'] = pd.cut(data['prev_duration'],
                                   bins=duration_bins).cat.codes
    data['neuroticism'] = data['neuroticism'].cat.codes
    data['prev_impairment'] = data['prev_impairment'] \
        .astype('category').cat.codes

    output_cols = ['exec_time',
                   'neuroticism', 'prev_impairment',
                   'prev_duration', 'transition']
    return PreprocessedData(data[output_cols].copy(),
                            Binner(bin_edges=neuroticism_bins),
                            Binner(bin_edges=impairment_bins),
                            Binner(bin_edges=duration_bins))


class ModelException(Exception):
    """
    Exception raised during model execution.
    """
    pass


class ExecutionTimeModel(abc.ABC):
    """
    Defines the general interface for execution time models.
    """

    @abc.abstractmethod
    def get_initial_step_execution_time(self) -> float:
        """
        Obtain an execution time for the first step in a task.

        Returns
        -------
        float
            An execution time value in seconds.
        """
        pass

    @abc.abstractmethod
    def get_execution_time(self, delay: float) -> float:
        """
        Obtain an execution time for a step N, N > 1.

        Parameters
        ----------
        delay
            Current measured delay in the system, in seconds.

        Returns
        -------
        float
            An execution time value in seconds.
        """
        pass

    @abc.abstractmethod
    def execution_time_iterator(self) -> ExecTimeIterator:
        """
        Utility method to obtain an iterator of execution times to use in a
        loop. Example use case, where `model` corresponds to an
        `ExecutionTimeModel` object::

            for exec_time in (model_iter := model.execution_time_iterator()):
                # do stuff
                # update delay before next iteration
                model_iter.set_delay(delay)

        Returns
        -------
        ExecTimeIterator
            An iterator for execution times.
        """
        pass


class ExecTimeIterator(Iterator[float]):
    """
    Utility class for iterating through execution times in a loop.

    Use case, if `model` corresponds to an `ExecutionTimeModel` object::

        for exec_time in (model_iter := model.execution_time_iterator()):
            # do stuff
            # update delay before next iteration
            model_iter.set_delay(delay)

    """

    def __init__(self, model: ExecutionTimeModel):
        self._delay = None

        def _generator() -> Generator[float]:
            # first call returns an execution time for an initial step
            yield model.get_initial_step_execution_time()

            # subsequent calls return values from non-initial steps
            while True:
                yield model.get_execution_time(self._get_delay())

        self._gen = _generator()

    def set_delay(self, value: Union[int, float]) -> None:
        """
        Set the current delay. Must be called before every iteration of this
        iterator except the first one.

        Parameters
        ----------
        value : float
            Current delay in seconds.

        """
        self._delay = float(value)

    def _get_delay(self) -> float:
        try:
            if self._delay is None:
                raise ModelException('Must call set_delay() between '
                                     'iterations of the execution '
                                     'time generator!')

            return self._delay
        finally:
            self._delay = None

    def __next__(self) -> float:
        return next(self._gen)

    def next(self, delay: Optional[Union[int, float]]) -> float:
        """
        Utility function which calls set_delay() before advancing the iteration.

        Parameters
        ----------
        delay : float
            Current delay in seconds.

        Returns
        -------
        float
            A value corresponding to the next execution time generated by the
            model.
        """
        self.set_delay(delay)
        return self.__next__()


class _EmpiricalExecutionTimeModel(ExecutionTimeModel):
    """
    Implementation of an execution time model which returns execution times
    sampled from the empirical distributions of the underlying data.
    """

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
        """
        Parameters
        ----------
        data
            A DataFrame indexed with pairs of (run/subject id, step sequence
            number) and with columns `prev_impairment`, `prev_duration`,
            and `transition`. Such as DataFrame can be obtained from the
            preprocess_data() function.
        neuro_level
            An integer corresponding to the binned level of neuroticism for
            this model.
        impair_binner
            A Binner object to bin delays into impairment levels.
        dur_binner
            A Binner object to bin duration values.
        """

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

    def execution_time_iterator(self) -> ExecTimeIterator[float]:
        return ExecTimeIterator(model=self)

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
    """
    Implementation of an execution time model which returns execution times
    sampled from theoretical distributions fitted to the underlying data.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 neuro_level: int,
                 impair_binner: Binner,
                 dur_binner: Binner,
                 distribution: stats.rv_continuous = stats.exponnorm):
        """
        Parameters
        ----------
        data
            A DataFrame indexed with pairs of (run/subject id, step sequence
            number) and with columns `prev_impairment`, `prev_duration`,
            and `transition`. Such as DataFrame can be obtained from the
            preprocess_data() function.
        neuro_level
            An integer corresponding to the binned level of neuroticism for
            this model.
        impair_binner
            A Binner object to bin delays into impairment levels.
        dur_binner
            A Binner object to bin duration values.
        distribution
            An scipy.stats.rv_continuous object corresponding to the
            distribution to fit to the empirical data. By default it
            corresponds to the Exponentially Modified Gaussian.
        """

        super(_TheoreticalExecutionTimeModel, self).__init__(data,
                                                             neuro_level,
                                                             impair_binner,
                                                             dur_binner)

        # at this point, the views have been populated with data according to
        # the binnings
        # now we fit distributions to each data view

        self._dists = {}
        for imp_dur_trans, df in self._data_views.items():
            # get the execution times, then fit the distribution to the samples

            exec_times = df['exec_time'].to_numpy()
            k, loc, scale = distribution.fit(exec_times)

            self._dists[imp_dur_trans] = \
                distribution.freeze(loc=loc, scale=scale, K=k)

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
    # TODO: document
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
