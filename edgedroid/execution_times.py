from __future__ import annotations

import abc
import enum
from collections import deque
from typing import Any, Dict, Iterator, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import arrays
from scipy import stats


class ModelException(Exception):
    """
    Exception raised during model execution.
    """

    pass


class Transition(str, enum.Enum):
    H2L = "Higher2Lower"
    L2H = "Lower2Higher"
    NONE = "NoTransition"


def preprocess_data(
    exec_time_data: pd.DataFrame,
    neuro_bins: arrays.IntervalArray | pd.IntervalIndex,
    impair_bins: arrays.IntervalArray | pd.IntervalIndex,
    duration_bins: arrays.IntervalArray | pd.IntervalIndex,
    transition_fade_distance: Optional[int] = None,
) -> pd.DataFrame:
    """
    Processes a DataFrame with raw execution time data into a DataFrame
    usable by the model.

    The argument DataFrame must be in order (of steps) and have the following
    columns:

    - run_id (categorical or int)
    - neuroticism (float)
    - exec_time (float)
    - delay (float)

    Parameters
    ----------
    exec_time_data
        Raw experimental data
    neuro_bins
        Bins to use for neuroticism values.
    impair_bins
        Bins to use for delay (impairment).
    duration_bins
        Bins to use for sequences of same impairment.
    transition_fade_distance
        Distance, in number of steps, from a transition after which to stop tagging
        steps wrt the latest transition.

    Returns
    -------
        A DataFrame.
    """

    for exp_col in ("run_id", "neuroticism", "exec_time", "delay"):
        if exp_col not in exec_time_data.columns:
            raise ModelException(f"Base dataframe missing column {exp_col}.")

    data = exec_time_data.copy()
    data["neuroticism"] = pd.cut(data["neuroticism"], pd.IntervalIndex(neuro_bins))
    data["impairment"] = pd.cut(data["delay"], pd.IntervalIndex(impair_bins))

    processed_dfs = deque()
    for run_id, df in data.groupby("run_id"):
        df = df.copy()
        df["next_exec_time"] = df["exec_time"].shift(-1)
        df["prev_impairment"] = df["impairment"].shift()
        df["transition"] = Transition.NONE.value

        # for each segment with the same impairment, count the number of steps
        # (starting from 1)
        diff_imp_groups = df.groupby(
            (df["impairment"].ne(df["prev_impairment"])).cumsum()
        )
        df["duration"] = diff_imp_groups.cumcount() + 1

        def tag_transition(df: pd.DataFrame) -> pd.DataFrame:
            # df is a chunk of the dataframe corresponding to a contiguos segment of
            # steps with the same impairment

            result = pd.DataFrame(index=df.index, columns=["transition"])
            result["transition"] = Transition.NONE.value

            mask = np.zeros(len(df.index), dtype=bool)
            mask[:transition_fade_distance] = True

            # hack to check if first element is none
            if df["prev_impairment"].astype(str).iloc[0] == "nan":
                # result[:transition_fade_distance] = Transition.NONE.value
                pass
            elif df["impairment"].iloc[0] > df["prev_impairment"].iloc[0]:
                result.loc[mask, "transition"] = Transition.L2H.value
            else:
                result.loc[mask, "transition"] = Transition.H2L.value

            # after a specific distance, the transition information "fades" and the
            # transition stops having an effect on the data
            # result.iloc[transition_fade_distance:]["transition"] =
            # Transition.NONE.value
            return result

        df["transition"] = diff_imp_groups.apply(tag_transition)
        df["duration"] = df.groupby(
            df["transition"].ne(df["transition"].shift()).cumsum()
        ).cumcount()

        df = df.drop(columns=["exec_time", "delay"])
        processed_dfs.append(df)

    data = pd.concat(processed_dfs, ignore_index=False)

    # coerce some types for proper functionality
    data["transition"] = data["transition"].astype("category")
    data["neuroticism"] = data["neuroticism"].astype(pd.IntervalDtype())
    data["impairment"] = data["impairment"].astype(pd.IntervalDtype())
    data["duration"] = pd.cut(data["duration"], pd.IntervalIndex(duration_bins)).astype(
        pd.IntervalDtype()
    )
    data = data.drop(columns="prev_impairment")

    return data


class ExecutionTimeModel(Iterator[float], metaclass=abc.ABCMeta):
    """
    Defines the general interface for execution time models.

    TODO: Define fallback behavior for missing data!
    """

    def __iter__(self):
        return self

    def __next__(self) -> float:
        return self.get_execution_time()

    @abc.abstractmethod
    def set_delay(self, delay: float | int) -> None:
        """
        Update the internal delay of this model.

        Parameters
        ----------
        delay
            A delay expressed in seconds.
        """
        pass

    @abc.abstractmethod
    def get_execution_time(self) -> float:
        """
        Obtain an execution time from this model.

        Returns
        -------
        float
            An execution time value in seconds.
        """
        pass

    @abc.abstractmethod
    def state_info(self) -> Dict[str, Any]:
        """
        Returns
        -------
        dict
            A dictionary containing debugging information about the internal
            stated of this model.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Resets the internal state to the starting state.
        """
        pass


class EmpiricalExecutionTimeModel(ExecutionTimeModel):
    """
    Implementation of an execution time model which returns execution times
    sampled from the empirical distributions of the underlying data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        neuroticism: float,
        transition_fade_distance: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        data
            A pd.DataFrame containing appropriate columns. Such a DataFrame
            can be obtained through the preprocess_data function in this module.
        neuroticism
            A normalized value of neuroticism for this model.
        """

        super().__init__()
        # first, we filter on neuroticism
        data = data[data["neuroticism"].array.contains(neuroticism)]
        self._neuroticism = neuroticism
        self._neuro_binned = data["neuroticism"].unique()[0]

        # def cleanup_execution_times(e: pd.Series) -> npt.NDArray:
        #     # clean up execution times by removing outliers identified with the
        #     # interquantile range method
        #
        #     q1 = e.quantile(0.25)
        #     q3 = e.quantile(0.75)
        #     iqr = q3 - q1
        #
        #     lower_fence = q1 - (iqr * 1.5)
        #     upper_fence = q3 + (iqr * 1.5)
        #
        #     return (
        #         e[(e >= lower_fence) & (e <= upper_fence)]
        #         .dropna()
        #         .to_numpy(dtype=np.float64)
        #     )

        # next, prepare views
        self._data_views = (
            data.groupby(
                ["impairment", "duration", "transition"], observed=True, dropna=True
            )["next_exec_time"]
            # .apply(cleanup_execution_times)
            .apply(lambda x: x.dropna().to_numpy()).to_dict()
        )

        # unique bins (interval arrays)
        self._duration_bins = data["duration"].unique()
        self._impairment_bins = data["impairment"].unique()

        # initial state
        self._fade_distance = (
            transition_fade_distance
            if transition_fade_distance is not None
            else float("inf")
        )
        self._duration = 0
        self._binned_duration = None
        self._impairment = None
        self._transition = Transition.NONE
        self.reset()

        # random state
        self._rng = np.random.default_rng()

    def reset(self) -> None:
        # initial state
        self._duration = 0
        self._binned_duration = self._duration_bins[
            self._duration_bins.contains(self._duration)
        ][0]
        self._impairment = self._impairment_bins[self._impairment_bins.contains(0)][0]
        self._transition = Transition.NONE

    def set_delay(self, delay: float | int) -> None:
        new_impairment = self._impairment_bins[self._impairment_bins.contains(delay)][0]

        if new_impairment > self._impairment:
            self._transition = Transition.L2H
            self._duration = 1
        elif new_impairment < self._impairment:
            self._transition = Transition.H2L
            self._duration = 1
        else:
            self._duration += 1
            if self._duration > self._fade_distance:
                self._duration = 1
                self._transition = Transition.NONE

        self._binned_duration = self._duration_bins[
            self._duration_bins.contains(self._duration)
        ][0]
        self._impairment = new_impairment

    def get_execution_time(self) -> float:
        # get the appropriate data view
        try:
            data: npt.NDArray = self._data_views[
                (self._impairment, self._binned_duration, self._transition.value)
            ]
        except KeyError:
            raise ModelException(f"No data for model state: {self.state_info()}!")

        # finally, sample from the data and return an execution time in seconds
        return self._rng.choice(data, replace=False)

    def state_info(self) -> Dict[str, Any]:
        return {
            "model neuroticism": self._neuroticism,
            "model neuroticism (binned)": self._neuro_binned,
            "latest impairment": self._impairment,
            "latest transition": self._transition.value,
            "current duration": self._duration,
            "current duration (binned)": self._binned_duration,
        }


class TheoreticalExecutionTimeModel(EmpiricalExecutionTimeModel):
    """
    Implementation of an execution time model which returns execution times
    sampled from theoretical distributions fitted to the underlying data.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        neuroticism: float,
        distribution: stats.rv_continuous = stats.exponnorm,
        transition_fade_distance: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        data
            A pd.DataFrame containing appropriate columns. Such a DataFrame
            can be obtained through the preprocess_data function in this module.
        neuroticism
            A normalized value of neuroticism for this model.
        distribution
            An scipy.stats.rv_continuous object corresponding to the
            distribution to fit to the empirical data. By default, it
            corresponds to the Exponentially Modified Gaussian.
        """

        super(TheoreticalExecutionTimeModel, self).__init__(
            data, neuroticism, transition_fade_distance
        )

        # at this point, the views have been populated with data according to
        # the binnings
        # now we fit distributions to each data view

        self._dists = {}
        for imp_dur_trans, exec_times in self._data_views.items():
            # get the execution times, then fit the distribution to the samples
            k, loc, scale = distribution.fit(exec_times)
            self._dists[imp_dur_trans] = distribution.freeze(loc=loc, scale=scale, K=k)

    def get_execution_time(self) -> float:
        # get the appropriate distribution
        try:
            dist = self._dists[
                (self._impairment, self._binned_duration, self._transition.value)
            ]
        except KeyError:
            raise ModelException(f"No data for model state: {self.state_info()}!")

        # finally, sample from the dist and return an execution time in seconds
        # note that we can't return negative values, so we'll end up changing the
        # distribution slightly by truncating at 0
        return max(dist.rvs(), 0)
