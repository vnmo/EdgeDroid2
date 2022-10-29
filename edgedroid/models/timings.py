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
from typing import Any, Dict, Iterator, Tuple, TypeVar

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
    # transition_fade_distance: Optional[int] = None,
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

    Returns
    -------
        A DataFrame.
    """

    proc_data = exec_time_data.copy()

    for col in ("run_id", "neuroticism", "exec_time", "ttf"):
        if col not in proc_data.columns:
            raise ModelException(f"Base data missing required column: {col}")

    proc_data["neuroticism_raw"] = proc_data["neuroticism"]
    proc_data["neuroticism"] = pd.cut(
        proc_data["neuroticism"], pd.IntervalIndex(neuro_bins)
    )

    processed_dfs = deque()
    for run_id, df in proc_data.groupby("run_id"):
        df = df.copy()
        df["ttf"] = df["ttf"].shift().fillna(0)

        df["impairment"] = pd.cut(df["ttf"], pd.IntervalIndex(impair_bins))
        df = df.rename(columns={"exec_time": "next_exec_time"})

        # df["next_exec_time"] = df["exec_time"].shift(-1)
        df["prev_impairment"] = df["impairment"].shift()
        # df["transition"] = Transition.NONE.value

        # for each segment with the same impairment, count the number of steps
        # (starting from 1)
        diff_imp_groups = df.groupby(
            (df["impairment"].ne(df["prev_impairment"])).cumsum()
        )
        df["duration"] = diff_imp_groups.cumcount() + 1

        df["transition"] = None
        df.loc[
            df["prev_impairment"] < df["impairment"], "transition"
        ] = Transition.L2H.value
        df.loc[
            df["prev_impairment"] > df["impairment"], "transition"
        ] = Transition.H2L.value

        df["transition"] = (
            df["transition"].fillna(method="ffill").fillna(Transition.NONE.value)
        )

        processed_dfs.append(df)

    proc_data = pd.concat(processed_dfs, ignore_index=False)

    # coerce some types for proper functionality
    proc_data["transition"] = proc_data["transition"].astype("category")
    proc_data["neuroticism"] = proc_data["neuroticism"].astype(pd.IntervalDtype(float))
    proc_data["impairment"] = proc_data["impairment"].astype(pd.IntervalDtype(float))
    proc_data["duration_raw"] = proc_data["duration"]
    proc_data["duration"] = pd.cut(
        proc_data["duration"], pd.IntervalIndex(duration_bins)
    ).astype(pd.IntervalDtype(float))
    proc_data = proc_data.drop(columns="prev_impairment")

    return proc_data


# workaround for typing methods of classes as returning the same type as the
# enclosing class, while also working for extending classes
TTimingModel = TypeVar("TTimingModel", bound="ExecutionTimeModel")


class ExecutionTimeModel(Iterator[float], metaclass=abc.ABCMeta):
    """
    Defines the general interface for execution time models.
    """

    @staticmethod
    def get_data() -> Tuple[
        pd.DataFrame,
        pd.arrays.IntervalArray,
        pd.arrays.IntervalArray,
        pd.arrays.IntervalArray,
    ]:
        import edgedroid.data as e_data

        return e_data.load_default_exec_time_data()

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
    def get_cdf_at_instant(self, instant: float):
        """
        Returns the value of the CDF for the execution time distribution of the
        current state of the model at the given instant.

        Parameters
        ----------
        instant: float

        Returns
        -------
        float
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


class ConstantExecutionTimeModel(ExecutionTimeModel):
    """
    Returns a constant execution time.
    """

    def __init__(self, execution_time_seconds: float):
        super(ConstantExecutionTimeModel, self).__init__()
        self._exec_time = execution_time_seconds

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "execution_time_seconds": float(self._exec_time),
        }

    def advance(self: TTimingModel, ttf: float | int) -> TTimingModel:
        # no-op
        return self

    def get_execution_time(self) -> float:
        return self._exec_time

    def get_expected_execution_time(self) -> float:
        return self.get_execution_time()

    def state_info(self) -> Dict[str, Any]:
        return {}

    def reset(self) -> None:
        # no-op
        pass

    def get_cdf_at_instant(self, instant: float):
        return float(instant > self._exec_time)


class NaiveExecutionTimeModel(ExecutionTimeModel):
    """
    Returns execution times sampled from a simple distribution.
    """

    def __init__(self):
        super(NaiveExecutionTimeModel, self).__init__()
        data, *_ = self.get_data()
        self._exec_times = data["exec_time"].to_numpy()
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

    def get_execution_time(self) -> float:
        return self._rng.choice(self._exec_times)

    def get_expected_execution_time(self) -> float:
        return self._exec_times.mean()

    def state_info(self) -> Dict[str, Any]:
        return {}

    def reset(self) -> None:
        # no-op
        pass

    def get_cdf_at_instant(self, instant: float) -> float:
        return self._exec_times[self._exec_times < instant].size / self._exec_times.size


class FittedNaiveExecutionTimeModel(NaiveExecutionTimeModel):
    def __init__(
        self,
        dist: stats.rv_continuous = stats.exponnorm,
    ):
        super(FittedNaiveExecutionTimeModel, self).__init__()

        *self._dist_args, self._loc, self._scale = dist.fit(self._exec_times)
        self._dist: stats.rv_continuous = dist.freeze(
            loc=self._loc, scale=self._scale, *self._dist_args
        )
        self._dist.random_state = self._rng

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "execution_time_seconds": {
                "distribution": self._dist.__class__.__name__,
                "loc": self._loc,
                "scale": self._scale,
                "other": list(self._dist_args),
            },
        }

    def advance(self: TTimingModel, ttf: float | int) -> TTimingModel:
        # no-op
        return self

    def get_execution_time(self) -> float:
        return float(self._dist.rvs())

    def get_expected_execution_time(self) -> float:
        return self._dist.expect()

    def state_info(self) -> Dict[str, Any]:
        return {}

    def reset(self) -> None:
        # no-op
        pass

    def get_cdf_at_instant(self, instant: float) -> float:
        return float(self._dist.cdf(instant))


class EmpiricalExecutionTimeModel(ExecutionTimeModel):
    """
    Implementation of an execution time model which returns execution times
    sampled from the empirical distributions of the underlying data.
    """

    def __init__(
        self,
        neuroticism: float | None,
        state_checks_enabled: bool = True,
    ):
        """
        Parameters
        ----------
        neuroticism
            A normalized value of neuroticism for this model.
        """

        super().__init__()
        data = preprocess_data(*self.get_data())

        # unique bins (interval arrays)
        self._duration_bins = data["duration"].unique()
        self._impairment_bins = data["impairment"].unique()
        self._neuro_bins = data["neuroticism"].unique()

        min_imp, max_imp = data["impairment"].min(), data["impairment"].max()
        self._min_dur = data["duration"].min()

        # first, we filter on neuroticism
        if neuroticism is not None:
            data = data[data["neuroticism"].array.contains(neuroticism)]
            self._neuro_binned = data["neuroticism"].iloc[0]
        else:
            self._neuro_binned = None

        self._neuroticism = neuroticism

        # next, prepare views
        self._data_views = {}
        for (imp, dur), df in data.groupby(["impairment", "duration"]):
            if (len(df.index) == 0) and state_checks_enabled:
                raise ModelException(
                    f"Combination of impairment {imp} and duration "
                    f"{dur} has no data!"
                )

            exec_times = df["next_exec_time"].to_numpy()

            # views for the beginning of the task, no transition
            # for that, we just use all the data for impairment and duration
            self._data_views[(imp, dur, Transition.NONE.value)] = exec_times

            if dur > self._min_dur:
                # transition only matters for the first level of duration
                self._data_views[(imp, dur, Transition.L2H.value)] = exec_times
                self._data_views[(imp, dur, Transition.H2L.value)] = exec_times
            else:
                for tran, tdf in df.groupby("transition"):
                    if tran == Transition.NONE.value:
                        continue
                    elif (len(tdf.index) == 0) and state_checks_enabled:
                        if (
                            ((imp == min_imp) and (tran == Transition.H2L.value))
                            or ((imp == max_imp) and (tran == Transition.L2H.value))
                            or ((imp != min_imp) and (imp != max_imp))
                        ):
                            raise ModelException(
                                f"Combination of impairment {imp}, duration "
                                f"{dur}, and transition {tran} has no data!"
                            )

                        # if we reach here it's a state which is impossible to reach,
                        # so we don't need to store anything
                    else:
                        self._data_views[(imp, dur, tran)] = tdf[
                            "next_exec_time"
                        ].to_numpy()

        self._imp_memory = deque()
        self._seq = 0
        self._ttf = 0.0
        self._transition = Transition.NONE.value
        # self.reset()

        # random state
        self._rng = np.random.default_rng()

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "neuroticism": (
                float(self._neuroticism) if self._neuroticism is not None else None
            ),
            "binned_neuroticism": (
                _serialize_interval(self._neuro_binned)
                if self._neuro_binned is not None
                else None
            ),
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
        self._transition = Transition.NONE.value
        self._rng = np.random.default_rng()

    def advance(self, ttf: float | int) -> TTimingModel:
        self._ttf = ttf
        new_imp = self._impairment_bins[self._impairment_bins.contains(ttf)][0]

        if len(self._imp_memory) < 1:
            # not enough steps to calculate a transition
            # must be first step
            self._seq = 0
            self._transition = Transition.NONE.value
        elif self._imp_memory[-1] < new_imp:
            self._imp_memory.clear()
            self._transition = Transition.L2H.value
        elif self._imp_memory[-1] > new_imp:
            self._imp_memory.clear()
            self._transition = Transition.H2L.value

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

        if binned_duration == self._min_dur:
            transition = Transition.NONE.value
        else:
            transition = self._transition

        try:
            return self._data_views[(self._imp_memory[-1], binned_duration, transition)]
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
                "transition": self._transition,
                "duration": binned_duration,
                "duration_raw": len(self._imp_memory),
            }
        except Exception as e:
            raise ModelException(
                f"Invalid model state. "
                f"Perhaps you forgot to advance() this model after "
                f"initialization?"
            ) from e

    def get_cdf_at_instant(self, instant: float) -> float:
        exec_times = self._get_data_for_current_state()
        return exec_times[exec_times < instant].size / exec_times.size


class TheoreticalExecutionTimeModel(EmpiricalExecutionTimeModel):
    """
    Implementation of an execution time model which returns execution times
    sampled from theoretical distributions fitted to the underlying data.
    """

    def __init__(
        self,
        neuroticism: float | None,
        distribution: stats.rv_continuous = stats.exponnorm,
        state_checks_enabled: bool = True,
    ):
        """
        Parameters
        ----------
        neuroticism
            A normalized value of neuroticism for this model.
        distribution
            An scipy.stats.rv_continuous object corresponding to the
            distribution to fit to the empirical data. By default, it
            corresponds to the Exponentially Modified Gaussian.
        """

        super(TheoreticalExecutionTimeModel, self).__init__(
            neuroticism=neuroticism,
            state_checks_enabled=state_checks_enabled,
        )

        # at this point, the views have been populated with data according to
        # the binnings
        # now we fit distributions to each data view

        self._dists = {}
        for imp_dur_trans, exec_times in self._data_views.items():
            # get the execution times, then fit the distribution to the samples
            *params, loc, scale = distribution.fit(exec_times)
            self._dists[imp_dur_trans] = distribution.freeze(
                *params,
                loc=loc,
                scale=scale,
            )

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

        if binned_duration == self._min_dur:
            transition = Transition.NONE.value
        else:
            transition = self._transition

        try:
            return self._dists[(self._imp_memory[-1], binned_duration, transition)]
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
        return self._get_dist_for_current_state().expect()

    def get_cdf_at_instant(self, instant: float) -> float:
        return float(self._get_dist_for_current_state().cdf(instant))


def _convolve_kernel(arr: pd.Series, kernel: npt.NDArray):
    index = arr.index
    arr = arr.to_numpy()
    arr = np.concatenate([np.zeros(kernel.size) + arr[0], arr])
    lkernel = np.concatenate([np.zeros(kernel.size - 1), kernel / kernel.sum()])
    result = np.convolve(arr, lkernel, "same")
    return pd.Series(result[kernel.size :], index=index)


class ExpKernelRollingTTFETModel(ExecutionTimeModel):
    @staticmethod
    def make_kernel(window: int, exp_factor: float = 0.7):
        kernel = np.zeros(window)
        for i in range(window):
            kernel[i] = np.exp(-exp_factor * i)

        return kernel / kernel.sum()

    def __init__(
        self,
        neuroticism: float | None,
        window: int = 12,
        ttf_levels: int = 7,
    ):

        data, neuro_bins, *_ = self.get_data()

        if neuroticism is not None:
            # bin neuroticism
            data["binned_neuro"] = pd.cut(
                data["neuroticism"], bins=pd.IntervalIndex(neuro_bins)
            ).astype(pd.IntervalDtype(float))
            data = data[data["binned_neuro"].array.contains(neuroticism)].copy()

        data["next_exec_time"] = data["exec_time"].shift(-1)
        data = data.dropna()

        # roll the ttfs
        self._kernel = self.make_kernel(window)
        data["rolling_ttf"] = data.groupby("run_id")["ttf"].apply(
            lambda arr: _convolve_kernel(arr, self._kernel)
        )
        _, ttf_bins = pd.qcut(data["rolling_ttf"], ttf_levels, retbins=True)
        ttf_bins[0], ttf_bins[-1] = -np.inf, np.inf
        self._ttf_bins = pd.IntervalIndex.from_breaks(ttf_bins, closed="right")
        data["binned_rolling_ttf"] = pd.cut(data["rolling_ttf"], bins=self._ttf_bins)

        # prepare views
        self._views: Dict[pd.Interval, npt.NDArray] = {}
        for binned_rolling_ttf, df in data.groupby("binned_rolling_ttf", observed=True):
            self._views[binned_rolling_ttf] = df["next_exec_time"].to_numpy()

        self._window = np.zeros(window, dtype=float)
        self._steps = 0
        self._neuroticism = neuroticism
        self._rng = np.random.default_rng()

    def advance(self: TTimingModel, ttf: float | int) -> TTimingModel:
        if self._steps == 0:
            self._window[:] = ttf
            self._steps += 1
        else:
            self._window = np.roll(self._window, shift=1)
            self._window[0] = ttf
        return self

    def _get_binned_ttf(self) -> pd.Interval:
        weighted_ttf = np.multiply(self._window, self._kernel).sum()
        return self._ttf_bins[self._ttf_bins.contains(weighted_ttf)][0]

    def get_execution_time(self) -> float:
        return self._rng.choice(self._views[self._get_binned_ttf()])

    def get_expected_execution_time(self) -> float:
        return self._views[self._get_binned_ttf()].mean()

    def state_info(self) -> Dict[str, Any]:
        return {
            "ttf_window": self._window,
            "weights": self._kernel,
            "weighted_ttf": np.multiply(self._window, self._kernel).sum(),
            "neuroticism": self._neuroticism,
            "steps": self._steps,
        }

    def reset(self) -> None:
        self._window = np.zeros(self._window.size, dtype=float)
        self._rng = np.random.default_rng()
        self._steps = 0

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "neuroticism": self._neuroticism,
            "window": self._window.size,
            "ttf_levels": len(self._ttf_bins),
        }

    def get_cdf_at_instant(self, instant: float) -> float:
        exec_times = self._views[self._get_binned_ttf()]
        return exec_times[exec_times < instant].size / exec_times.size


class DistExpKernelRollingTTFETModel(ExpKernelRollingTTFETModel):
    def __init__(
        self,
        neuroticism: float | None,
        dist: stats.rv_continuous = stats.exponnorm,
        window: int = 8,
        ttf_levels: int = 7,
    ):
        super(DistExpKernelRollingTTFETModel, self).__init__(
            neuroticism=neuroticism, window=window, ttf_levels=ttf_levels
        )

        self._dists: Dict[pd.Interval, stats.rv_continuous] = {}
        for ttf_bin, exec_times in self._views.items():
            *params, loc, scale = dist.fit(exec_times)
            self._dists[ttf_bin] = dist.freeze(
                *params,
                loc=loc,
                scale=scale,
            )

        self._distribution = dist

    def get_execution_time(self) -> float:
        return max(self._dists[self._get_binned_ttf()].rvs(), 0.0)

    def get_expected_execution_time(self) -> float:
        return self._dists[self._get_binned_ttf()].expect()

    def get_model_params(self) -> Dict[str, Any]:
        params = super(DistExpKernelRollingTTFETModel, self).get_model_params()
        params["distribution"] = self._distribution.name
        return params

    def get_cdf_at_instant(self, instant: float) -> float:
        return float(self._dists[self._get_binned_ttf()].cdf(instant))
