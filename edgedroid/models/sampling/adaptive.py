from __future__ import annotations

import abc
import itertools
import time
from collections import deque
from typing import Iterator, Type

import numpy as np
import pandas as pd
from numpy import typing as npt

from .base import BaseFrameSamplingModel, FrameSample, TBaseSampling
from ..timings import ExecutionTimeModel
from ... import data as e_data


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


class AperiodicFrameSamplingModel(BaseAdaptiveFrameSamplingModel):
    """
    Implements Vishnu's aperiodic sampling.
    """

    @classmethod
    def from_default_data(
        cls: Type[TBaseSampling],
        execution_time_model: ExecutionTimeModel,
        delay_cost_window: int = 5,
        beta: float = 1.0,
        init_nettime_guess=0.3,
        proctime: float = 0.0,
        *args,
        **kwargs,
    ) -> TBaseSampling:
        probs = e_data.load_default_frame_probabilities()
        return cls(
            probabilities=probs,
            execution_time_model=execution_time_model,
            success_tag="success",
            delay_cost_window=delay_cost_window,
            beta=beta,
            init_nettime_guess=init_nettime_guess,
            proctime=proctime,
        )

    def __init__(
        self,
        probabilities: pd.DataFrame,
        execution_time_model: ExecutionTimeModel,
        success_tag: str = "success",
        delay_cost_window: int = 5,
        beta: float = 1.0,
        init_nettime_guess=0.3,
        proctime: float = 0.0,
    ):
        """

        Parameters
        ----------
        probabilities
            Frame probabilities
        execution_time_model
            An execution time model to predict the next steps execution time.
        success_tag

        # init_network_time_guess_seconds
        #     Initial guess for the network time.
        # processing_time_seconds
        #     Factor, expressed in seconds, that is subtracted from frame round-trip
        #     times, representing the time processing took on the backend.
        # idle_factor
        #     Estimated idle power consumption of the client.
        # busy_factor
        #     Estimated communication power consumption of the client.
        # network_time_window
        #     Size of the network time window, in number of samples, used to calculate
        #     the average network time at each step.
        """
        super(AperiodicFrameSamplingModel, self).__init__(
            probabilities=probabilities,
            execution_time_model=execution_time_model,
            success_tag=success_tag,
        )

        # self._initial_nt_guess = init_network_time_guess_seconds
        # self._network_times = deque()
        # TODO: defaults are magic numbers

        self._delay_costs = deque(
            [init_nettime_guess],
            maxlen=delay_cost_window,
        )

        self._processing_time = proctime
        self._beta = beta

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

        # TODO: check how often devolves into zero-wait

        alpha = float(np.mean(self._delay_costs))

        for i, target_instant in enumerate(
            _aperiodic_instant_iterator(
                mu=self._timing_model.get_expected_execution_time(),
                alpha=alpha,
                beta=self._beta,
                # alpha=self._current_rtt_mean *
                # (self._busy_factor - self._idle_factor),
                # beta=self._idle_factor,
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
                    "beta": self._beta,
                    "ttf": ttf,
                    "target_instant": target_instant,
                    "late": late,
                    "delay_cost_window": self._delay_costs.maxlen,
                },
            )
            dt = (time.monotonic() - step_start) - instant

            if instant > target_time:
                num_samples = i + 1
                self._delay_costs.append(
                    max(dt - self._processing_time, 0.0) / num_samples
                )

                # self._current_rtt_mean = (
                #     np.mean(step_rtts) if len(step_rtts) > 0 else self._
                #     current_rtt_mean
                # )
                break

            # only add frame rtt to collection if it's not a transition frame
            # step_rtts.append(max(dt - self._processing_time, 0.0))
