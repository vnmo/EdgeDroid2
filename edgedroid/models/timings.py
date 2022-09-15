#  Copyright (c) 2022 Manuel Olguín Muñoz <molguin@kth.se>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import abc
import copy
import enum
from collections import deque
from typing import Any, Dict, Iterator, Optional, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import arrays
from scipy import stats


def _serialize_interval(interval: pd.Interval) -> Dict[str, float | bool]:
    left_open = interval.open_left
    right_open = interval.open_right

    if left_open and right_open:
        closed = "neither"
    elif left_open:
        closed = "right"
    elif right_open:
        closed = "left"
    else:
        closed = "both"

    return {
        "left": float(interval.left),
        "right": float(interval.right),
        "closed": closed,
    }


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
    - ttf (float)

    Parameters
    ----------
    exec_time_data
        Raw experimental data
    neuro_bins
        Bins to use for neuroticism values.
    impair_bins
        Bins to use for time-to-feedback (impairment).
    duration_bins
        Bins to use for sequences of same impairment.
    transition_fade_distance
        Distance, in number of steps, from a transition after which to stop tagging
        steps wrt the latest transition.

    Returns
    -------
        A DataFrame.
    """

    data = exec_time_data.copy()

    for col in ("run_id", "neuroticism", "exec_time", "ttf"):
        if col not in data.columns:
            raise ModelException(f"Base data missing required column: {col}")

    data["neuroticism_raw"] = data["neuroticism"]
    data["neuroticism"] = pd.cut(data["neuroticism"], pd.IntervalIndex(neuro_bins))

    processed_dfs = deque()
    for run_id, df in data.groupby("run_id"):
        # df = df.copy()
        df = df.copy()
        df["ttf"] = df["ttf"].shift().fillna(0)

        df["impairment"] = pd.cut(df["ttf"], pd.IntervalIndex(impair_bins))
        df = df.rename(columns={"exec_time": "next_exec_time"})

        # df["next_exec_time"] = df["exec_time"].shift(-1)
        df["prev_impairment"] = df["impairment"].shift()
        df["transition"] = Transition.NONE.value

        # for each segment with the same impairment, count the number of steps
        # (starting from 1)
        diff_imp_groups = df.groupby(
            (df["impairment"].ne(df["prev_impairment"])).cumsum()
        )
        df["duration"] = diff_imp_groups.cumcount() + 1

        def tag_transition(df: pd.DataFrame) -> pd.DataFrame:
            # df is a chunk of the dataframe corresponding to a contiguous segment of
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
        df["duration"] = (
            df.groupby(
                (
                    df["transition"].ne(df["transition"].shift())
                    | df["impairment"].ne(df["prev_impairment"])
                ).cumsum()
            ).cumcount()
            + 1
        )

        processed_dfs.append(df)

    data = pd.concat(processed_dfs, ignore_index=False)

    # coerce some types for proper functionality
    data["transition"] = data["transition"].astype("category")
    data["neuroticism"] = data["neuroticism"].astype(pd.IntervalDtype())
    data["impairment"] = data["impairment"].astype(pd.IntervalDtype())
    data["duration_raw"] = data["duration"]
    data["duration"] = pd.cut(data["duration"], pd.IntervalIndex(duration_bins)).astype(
        pd.IntervalDtype()
    )
    data = data.drop(columns="prev_impairment")

    return data


def winsorize_series(e: pd.Series) -> npt.NDArray:
    # clean up execution times by setting
    # x = 5th percentile, for all values < 5th percentile
    # x = 95th percentile, for all values > 95th percentile

    e = e.dropna().to_numpy(dtype=np.float64)

    percs = np.percentile(e, [5, 95])
    mask5 = e < percs[0]
    mask95 = e > percs[1]

    e[mask5] = percs[0]
    e[mask95] = percs[1]
    return e


# workaround for typing methods of classes as returning the same type as the
# enclosing class, while also working for extending classes
TTimingModel = TypeVar("TTimingModel", bound="ExecutionTimeModel")


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
    def advance(self: TTimingModel, ttf: float | int) -> TTimingModel:
        """
        Update the internal TTF of this model and advance the internal state.

        Parameters
        ----------
        ttf
            Time-to-feedback of the previous step, expressed in seconds.
        """
        return self

    @abc.abstractmethod
    def get_impairment_score(self) -> float:
        """
        Returns
        -------
        float
            A normalized "impairment" score for this model, where 0 corresponds to
            completely unimpaired and 1.0 to completely impaired.

            This score is derived from the maximum and minimum mean execution times
            this model can produce. The mean of the current state is then normalized
            against these values to produce the score.
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
    def get_expected_execution_time(self) -> float:
        """
        Returns the *expected* execution time for the current state of the model.

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

    def copy(self: TTimingModel) -> TTimingModel:
        """
        Returns a (deep) copy of this model.
        """
        return copy.deepcopy(self)

    def fresh_copy(self: TTimingModel) -> TTimingModel:
        """
        Returns a (deep) copy of this model, reset to its initial state.
        """
        model = self.copy()
        model.reset()
        return model

    @abc.abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        pass


class NaiveExecutionTimeModel(ExecutionTimeModel):
    """
    Returns a constant execution time.
    """

    @classmethod
    def from_default_data(cls) -> ExecutionTimeModel:
        """
        Builds a naive model from the default data, using the average execution time
        of the best-case (unimpaired) steps as the execution time for the model.

        Returns
        -------
        ExecutionTimeModel
            A naive execution time model.
        """

        # TODO: should this model be more complex? Maybe use a (simple) distribution?

        from .. import data as e_data

        data = preprocess_data(
            *e_data.load_default_exec_time_data(),
            transition_fade_distance=4,
        )

        # only grab execution times for steps that
        # 1. correspond to an "unimpaired" state
        # 2. are not marked as being close to a transition.
        data = data[
            (data["impairment"] == data["impairment"].min())
            & (data["transition"] == Transition.NONE.value)
        ]

        # clean outliers
        exec_times = winsorize_series(data["next_exec_time"])
        return cls(exec_times.mean())

    def __init__(self, execution_time_seconds: float):
        super(NaiveExecutionTimeModel, self).__init__()
        self._exec_time = execution_time_seconds

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "execution_time_seconds": float(self._exec_time),
        }

    def advance(self: TTimingModel, ttf: float | int) -> TTimingModel:
        # no-op
        return self

    def get_impairment_score(self) -> float:
        return 0.0

    def get_execution_time(self) -> float:
        return self._exec_time

    def get_expected_execution_time(self) -> float:
        return self.get_execution_time()

    def state_info(self) -> Dict[str, Any]:
        return {}

    def reset(self) -> None:
        # no-op
        pass


class ProbabilisticNaiveExecutionTimeModel(ExecutionTimeModel):
    """
    Returns execution times sampled from a simple distribution.
    """

    @classmethod
    def from_default_data(cls) -> ExecutionTimeModel:
        """
        Builds a naive model from the default data.

        Returns
        -------
        ExecutionTimeModel
            A naive execution time model.
        """
        from .. import data as e_data

        data, *_ = e_data.load_default_exec_time_data()
        return cls(data["exec_time"].to_numpy())

    def __init__(self, execution_times: npt.NDArray):
        super(ProbabilisticNaiveExecutionTimeModel, self).__init__()
        self._exec_times = execution_times
        self._rng = np.random.default_rng()

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "execution_time_seconds": {
                "mean": float(self._exec_times.mean()),
                "std": float(self._exec_times.std()),
            }
        }

    def advance(self: TTimingModel, ttf: float | int) -> TTimingModel:
        # no-op
        return self

    def get_impairment_score(self) -> float:
        return 0.0

    def get_execution_time(self) -> float:
        return self._rng.choice(self._exec_times)

    def get_expected_execution_time(self) -> float:
        return self._exec_times.mean()

    def state_info(self) -> Dict[str, Any]:
        return {}

    def reset(self) -> None:
        # no-op
        pass


class EmpiricalExecutionTimeModel(ExecutionTimeModel):
    """
    Implementation of an execution time model which returns execution times
    sampled from the empirical distributions of the underlying data.
    """

    @classmethod
    def from_default_data(
        cls, neuroticism: float, *args, transition_fade_distance: int = 8, **kwargs
    ) -> ExecutionTimeModel:
        from .. import data as e_data

        data = preprocess_data(
            *e_data.load_default_exec_time_data(),
            transition_fade_distance=transition_fade_distance,
        )

        return cls(
            *args,
            data=data,
            neuroticism=neuroticism,
            transition_fade_distance=transition_fade_distance,
            **kwargs,
        )

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
        self._neuro_bins = data["neuroticism"].unique()

        # first, we filter on neuroticism
        data = data[data["neuroticism"].array.contains(neuroticism)]
        self._neuroticism = neuroticism
        self._neuro_binned = data["neuroticism"].iloc[0]

        # next, prepare views
        self._data_views = (
            data.groupby(
                ["impairment", "duration", "transition"], observed=True, dropna=True
            )["next_exec_time"]
            .apply(winsorize_series)
            .to_dict()
            # .apply(lambda x: x.dropna().to_numpy()).to_dict()
        )

        # calculate imp score range
        _min_exec_time_mean = np.inf
        _max_exec_time_mean = -np.inf
        for _, exec_times in self._data_views.items():
            current_mean = exec_times.mean()
            _min_exec_time_mean = np.min((_min_exec_time_mean, current_mean))
            _max_exec_time_mean = np.max((_max_exec_time_mean, current_mean))

        self._max_score = _max_exec_time_mean - _min_exec_time_mean
        self._score_offset = _min_exec_time_mean

        # unique bins (interval arrays)
        self._duration_bins = data["duration"].unique()
        self._impairment_bins = data["impairment"].unique()

        # initial state
        self._fade_distance = (
            transition_fade_distance
            if transition_fade_distance is not None
            else float("inf")
        )

        self._imp_memory = deque()
        self._seq = 0
        self._ttf = 0.0
        self._transition = None
        self.reset()

        # random state
        self._rng = np.random.default_rng()

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "neuroticism": float(self._neuroticism),
            "binned_neuroticism": _serialize_interval(self._neuro_binned),
            "fade_distance": int(self._fade_distance),
            "neuro_bins": [_serialize_interval(iv) for iv in self._neuro_bins],
            "impairment_bins": [
                _serialize_interval(iv) for iv in self._impairment_bins
            ],
            "duration_bins": [_serialize_interval(iv) for iv in self._duration_bins],
        }

    def copy(self: TTimingModel) -> TTimingModel:
        model_copy = super(EmpiricalExecutionTimeModel, self).copy()

        # make sure to reinit the random number generator
        model_copy._rng = np.random.default_rng()
        return model_copy

    def reset(self) -> None:
        # initial state
        self._imp_memory.clear()
        self._seq = 0
        self._ttf = 0.0
        self._transition = None

    def advance(self, ttf: float | int) -> TTimingModel:
        self._ttf = ttf
        new_imp = self._impairment_bins[self._impairment_bins.contains(ttf)][0]

        if len(self._imp_memory) < 1:
            # not enough steps to calculate a transition
            # must be first step
            self._seq = 0
            self._transition = Transition.NONE
        elif self._imp_memory[-1] < new_imp:
            self._imp_memory.clear()
            self._transition = Transition.L2H
        elif self._imp_memory[-1] > new_imp:
            self._imp_memory.clear()
            self._transition = Transition.H2L
        elif (
            len(self._imp_memory) + 1 > self._fade_distance
            and self._transition != Transition.NONE
        ):
            self._imp_memory.clear()
            self._transition = Transition.NONE

        self._seq += 1
        self._imp_memory.append(new_imp)
        # self._binned_duration = self._duration_bins[
        #     self._duration_bins.contains(self._duration)
        # ][0]

        return self

    def _get_data_for_current_state(self) -> npt.NDArray:
        # get the appropriate data view
        binned_duration = self._duration_bins[
            self._duration_bins.contains(len(self._imp_memory))
        ][0]

        try:
            return self._data_views[
                (self._imp_memory[-1], binned_duration, self._transition.value)
            ]
        except KeyError:
            raise ModelException(
                f"No data for model state: {self.state_info()}! "
                f"Perhaps you forgot to advance() this model after "
                f"initialization?"
            )

    def get_execution_time(self) -> float:
        # sample from the data and return an execution time in seconds
        return self._rng.choice(self._get_data_for_current_state(), replace=False)

    def get_expected_execution_time(self) -> float:
        return self._get_data_for_current_state().mean()

    def get_impairment_score(self) -> float:
        current_mean = self._get_data_for_current_state().mean()
        return (current_mean - self._score_offset) / self._max_score

    def state_info(self) -> Dict[str, Any]:
        try:
            binned_duration = self._duration_bins[
                self._duration_bins.contains(len(self._imp_memory))
            ][0]

            return {
                "seq": self._seq,
                "neuroticism": self._neuro_binned,
                "neuroticism_raw": self._neuroticism,
                "ttf": self._ttf,
                "impairment": self._imp_memory[-1],
                "transition": self._transition.value,
                "duration": binned_duration,
                "duration_raw": len(self._imp_memory),
            }
        except Exception as e:
            raise ModelException(
                f"Invalid model state. "
                f"Perhaps you forgot to advance() this model after "
                f"initialization?"
            ) from e


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

        # calculate imp score range, based on distributions
        _min_exec_time_mean = np.inf
        _max_exec_time_mean = -np.inf
        for _, dist in self._dists.items():
            current_mean = dist.mean()
            _min_exec_time_mean = np.min((_min_exec_time_mean, current_mean))
            _max_exec_time_mean = np.max((_max_exec_time_mean, current_mean))

        self._max_score = _max_exec_time_mean - _min_exec_time_mean
        self._score_offset = _min_exec_time_mean
        self._distribution = distribution

    def get_model_params(self) -> Dict[str, Any]:
        params = super(TheoreticalExecutionTimeModel, self).get_model_params()
        params["distribution"] = str(self._distribution.name)
        return params

    def _get_dist_for_current_state(self) -> stats.rv_continuous:
        # get the appropriate distribution
        binned_duration = self._duration_bins[
            self._duration_bins.contains(len(self._imp_memory))
        ][0]

        try:
            return self._dists[
                (self._imp_memory[-1], binned_duration, self._transition.value)
            ]
        except KeyError:
            raise ModelException(
                f"No data for model state: {self.state_info()}! "
                f"Perhaps you forgot to advance() this model after "
                f"initialization?"
            )

    def get_execution_time(self) -> float:
        # sample from the dist and return an execution time in seconds
        # note that we can't return negative values, so we'll end up changing the
        # distribution slightly by truncating at 0
        return max(self._get_dist_for_current_state().rvs(), 0)

    def get_expected_execution_time(self) -> float:
        return self._get_dist_for_current_state().mean()

    def get_impairment_score(self) -> float:
        current_mean = self._get_dist_for_current_state().mean()
        return (current_mean - self._score_offset) / self._max_score
