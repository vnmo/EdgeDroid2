from __future__ import annotations

import abc
import itertools
import time
from collections import deque
from typing import Iterator, Sequence, Type

import numpy as np
import pandas as pd
from numpy import typing as npt

from .base import BaseFrameSamplingModel, FrameSample, TBaseSampling
from ..timings import ExecutionTimeModel


# from ... import data as e_data


def _aperiodic_instant_iterator(
    mu: float,
    alpha: float,
    beta: float,
    n_offset: int = 1,
) -> Iterator[float]:
    """
    Iterates over (quasi) optimal sampling instants for a step, assuming execution
    times sampled from a Rayleigh distribution with expected value mu.

    Parameters
    ----------
    mu
        Expected value (mean) of the Rayleigh distribution from which execution times
        are drawn.
    alpha
        Penalty parameter.
    beta
        Penalty parameter.
    n_offset
        Instant from which to start yielding values.

    Yields
    ------
    float
        Sampling instants, expressed in seconds relative to the beginning of the step.
    """

    sigma = np.sqrt(np.divide(2, np.pi)) * mu
    const_factor = np.float_power(
        3 * sigma * np.sqrt(np.divide(alpha, 2 * beta)),
        np.divide(2, 3),
    )

    if np.isnan(sigma) or np.isnan(const_factor):
        raise RuntimeError(
            f"NaN values in sampling instant calculation! "
            f"{mu=} "
            f"{alpha=} "
            f"{beta=} "
            f"{sigma=} "
            f"{const_factor=} "
        )

    for n in itertools.count(start=n_offset):
        yield const_factor * np.float_power(n, np.divide(2, 3))


def _aperiodic_sampling_instants(
    # t1_seconds: float,
    mu: float,
    alpha: float,
    beta: float,
    dt_thresh_s: float = 1 / 1000,  # 1 ms
    ddt_thresh_s: float = 1 / 10000,
) -> npt.NDArray:
    """
    Generates (quasi) optimal sampling intervals for a step, assuming execution times
    sampled from a Rayleigh distribution with expected value mu. Always returns at
    least a first sampling instant at t = 0, and a second instant t1.

    Parameters
    ----------
    mu
        Expected value (mean) of the Rayleigh distribution from which execution times
        are drawn.
    alpha
        Penalty parameter.
    beta
        Penalty parameter.
    dt_thresh_s
        Threshold for the minimum interval between successive samples.
        Once (t_new - t_previous) is less than this value, stop generating new instants.
    ddt_thresh_s
        Threshold for the minimum delta between successive deltas (i.e. second
        derivative of the time instants).

    Returns
    -------
    npt.NDArray
        A numpy array of sampling instants.
    """

    instant_iter = _aperiodic_instant_iterator(
        mu=mu, alpha=alpha, beta=beta, n_offset=0
    )

    t0 = next(instant_iter)
    t1 = next(instant_iter)
    instants = deque([t0, t1])

    prev_dt = t1 - t0

    for tn in instant_iter:
        dt = tn - instants[-1]
        if (dt < dt_thresh_s) or ((prev_dt - dt) < ddt_thresh_s):
            return np.array(instants, dtype=float)

        prev_dt = dt
        instants.append(tn)


class BaseAdaptiveFrameSamplingModel(BaseFrameSamplingModel, metaclass=abc.ABCMeta):
    @classmethod
    def from_default_data(
        cls: Type[TBaseSampling],
        execution_time_model: ExecutionTimeModel,
        *args,
        **kwargs,
    ) -> TBaseSampling:
        from ... import data as e_data

        probs = e_data.load_default_frame_probabilities()
        return cls(
            probabilities=probs,
            execution_time_model=execution_time_model,
            success_tag="success",
        )

    def __init__(
        self,
        probabilities: pd.DataFrame,
        execution_time_model: ExecutionTimeModel,
        success_tag: str = "success",
    ):
        super(BaseAdaptiveFrameSamplingModel, self).__init__(
            probabilities=probabilities,
            success_tag=success_tag,
        )

        self._timing_model = execution_time_model.copy()


class BaseAperiodicFrameSamplingModel(BaseAdaptiveFrameSamplingModel, abc.ABC):
    @abc.abstractmethod
    def get_alpha(self) -> float:
        ...

    @abc.abstractmethod
    def get_beta(self) -> float:
        ...

    @abc.abstractmethod
    def record_rtts(self, rtts: Sequence[float]) -> None:
        ...

    def step_iterator(
        self,
        target_time: float,
        ttf: float,
        # infinite: bool = False,
    ) -> Iterator[FrameSample]:

        step_start = time.monotonic()
        # step_rtts = deque()

        # Tc = (
        #     np.mean(self._network_times)
        #     if len(self._network_times) > 0
        #     else self._initial_nt_guess
        # )
        self._timing_model.advance(ttf)

        alpha = self.get_alpha()
        beta = self.get_beta()
        rtts = deque()

        for i, target_instant in enumerate(
            _aperiodic_instant_iterator(
                mu=self._timing_model.get_expected_execution_time(),
                alpha=alpha,
                beta=beta,
            )
        ):
            try:
                time.sleep(target_instant - (time.monotonic() - step_start))
                late = False
            except ValueError:
                # missed the sampling instant
                late = True

            instant = time.monotonic() - step_start
            yield FrameSample(
                seq=i + 1,
                sample_tag=self.get_frame_at_instant(instant, target_time),
                instant=instant,
                extra={
                    "alpha": alpha,
                    "beta": beta,
                    "ttf": ttf,
                    "target_instant": target_instant,
                    "late": late,
                },
            )
            rtts.append((time.monotonic() - step_start) - instant)

            if instant > target_time:
                self.record_rtts(rtts)
                break


class AperiodicFrameSamplingModel(BaseAperiodicFrameSamplingModel):
    """
    Implements Vishnu's aperiodic sampling, optimized for time.
    """

    @classmethod
    def from_default_data(
        cls: Type[TBaseSampling],
        execution_time_model: ExecutionTimeModel,
        delay_cost_window: int = 5,
        beta: float = 1.0,
        init_rtt_guess=0.32,
        proctime: float = 0.3,
        *args,
        **kwargs,
    ) -> TBaseSampling:
        from ... import data as e_data

        probs = e_data.load_default_frame_probabilities()
        return cls(
            probabilities=probs,
            execution_time_model=execution_time_model,
            success_tag="success",
            delay_cost_window=delay_cost_window,
            beta=beta,
            init_rtt_guess=init_rtt_guess,
            proctime=proctime,
        )

    def __init__(
        self,
        probabilities: pd.DataFrame,
        execution_time_model: ExecutionTimeModel,
        success_tag: str = "success",
        delay_cost_window: int = 5,
        beta: float = 1.0,
        init_rtt_guess=0.32,
        proctime: float = 0.3,
    ):
        """

        Parameters
        ----------
        probabilities
            Frame probabilities
        execution_time_model
            An execution time model to predict the next steps execution time.
        success_tag
        """
        super(AperiodicFrameSamplingModel, self).__init__(
            probabilities=probabilities,
            execution_time_model=execution_time_model,
            success_tag=success_tag,
        )

        # self._initial_nt_guess = init_network_time_guess_seconds
        # self._network_times = deque()
        # TODO: defaults are magic numbers

        self._alpha = 0.0
        self._beta = beta
        self._processing_time = proctime

        self._delay_costs = deque(maxlen=delay_cost_window)
        self.record_rtts([init_rtt_guess])

    def get_beta(self) -> float:
        return self._beta

    def get_alpha(self) -> float:
        return self._alpha

    def record_rtts(self, rtts: Sequence[float]) -> None:
        self._delay_costs.append(max(rtts[-1] - self._processing_time, 0.0) / len(rtts))
        self._alpha = float(np.mean(self._delay_costs))


class AperiodicPowerFrameSamplingModel(BaseAperiodicFrameSamplingModel):
    # TODO: add to command line

    @classmethod
    def from_default_data(
        cls: Type[TBaseSampling],
        execution_time_model: ExecutionTimeModel,
        success_tag: str = "success",
        idle_power_w: float = 0.015,
        comm_power_w: float = 0.045,
        init_rtt_guess=0.32,
        proctime_s: float = 0.3,
        rtt_window_size: int = 10,
        *args,
        **kwargs,
    ) -> TBaseSampling:
        from ... import data as e_data

        probs = e_data.load_default_frame_probabilities()
        return cls(
            probabilities=probs,
            execution_time_model=execution_time_model,
            success_tag=success_tag,
            idle_power_w=idle_power_w,
            comm_power_w=comm_power_w,
            init_rtt_guess=init_rtt_guess,
            proctime_s=proctime_s,
            rtt_window_size=rtt_window_size,
        )

    def __init__(
        self,
        probabilities: pd.DataFrame,
        execution_time_model: ExecutionTimeModel,
        success_tag: str = "success",
        idle_power_w: float = 0.015,
        comm_power_w: float = 0.045,
        init_rtt_guess=0.32,
        proctime_s: float = 0.3,
        rtt_window_size: int = 10,
    ):
        super(BaseAperiodicFrameSamplingModel, self).__init__(
            probabilities=probabilities,
            execution_time_model=execution_time_model,
            success_tag=success_tag,
        )

        self._P0 = idle_power_w
        self._Pc = comm_power_w
        self._tc = 0.0
        self._processing_time = proctime_s

        self._rtt_window = deque(maxlen=rtt_window_size)
        self.record_rtts([init_rtt_guess])

    def record_rtts(self, rtts: Sequence[float]) -> None:
        for rtt in rtts:
            self._rtt_window.append(max(rtt - self._processing_time, 0.0))
        self._tc = float(np.mean(self._rtt_window))

    def get_alpha(self) -> float:
        return self._tc * (self._Pc - self._P0)

    def get_beta(self) -> float:
        return self._P0
