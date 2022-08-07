from __future__ import annotations

import abc
import itertools
import time
from collections import deque
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from numpy import typing as npt

from .base import BaseFrameSamplingModel
from ..timings import ExecutionTimeModel


def _aperiodic_instant_iterator(
    mu: float,
    alpha: float,
    beta: float,
    n_offset: int = 1,
) -> Iterator[float]:
    """
    Iterates over (quasi) optimal sampling intervals for a step, assuming execution
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


# def _periodic_sampling_interval(
#     mu: float,
#     alpha: float,
#     beta: float,
#     min_possible_exec_time: float = 0.5,
# ) -> float:
#     sigma = np.sqrt(np.divide(2, np.pi)) * mu
#     K_max = int(np.ceil(100.0 * sigma / min_possible_exec_time))
#
#     def _energy_penalty(Ts: float) -> float:
#         Ts_factor = alpha + (beta * Ts)
#
#         s2_sigma = np.sqrt(2) * sigma
#
#         def exponent(k: int) -> float:
#             return -np.square(np.divide(k * Ts, s2_sigma))
#
#         sum_factor = sum([np.exp(exponent(k)) for k in range(K_max + 1)])
#
#         offset = beta * sigma * np.sqrt(np.divide(np.pi, 2))
#
#         return Ts_factor * sum_factor - offset
#
#     return scipy.optimize.minimize_scalar(_energy_penalty).x
#
#
# def _energy_penalty_with_offset(
#     Ts: float,
#     delta: float,
#     alpha: float,
#     beta: float,
#     sigma: float,
#     K_max: int,
# ) -> float:
#     Ts_factor = alpha + (beta * Ts)
#     s2_sigma = np.sqrt(2) * sigma
#
#     def exponent(k: int) -> float:
#         return -np.square(np.divide((k * Ts) + delta, s2_sigma))
#
#     sum_factor = sum([np.exp(exponent(k)) for k in range(K_max + 1)])
#
#     return (
#         (Ts_factor * sum_factor)
#         - (beta * sigma * np.sqrt(np.divide(np.pi, 2)))
#         + (beta * delta)
#         + alpha
#     )
#
#
# def _periodic_sampling_interval_with_offset(
#     mu: float,
#     alpha: float,
#     beta: float,
#     Ts_range: Tuple[float, float],
#     sampling_precision: float = 0.01,
#     min_possible_exec_time: float = 0.5,
#     delta_range: Optional[Tuple[float, float]] = None,
# ) -> Tuple[float, float]:
#     Ts_values = np.linspace(
#         start=Ts_range[0],
#         stop=Ts_range[1],
#         endpoint=True,
#         num=int(np.rint((Ts_range[1] - Ts_range[0]) / sampling_precision)),
#     )
#
#     if delta_range is None:
#         delta_values = Ts_values.copy()
#     else:
#         delta_values = np.linspace(
#             start=delta_range[0],
#             stop=delta_range[1],
#             endpoint=True,
#             num=int(np.rint((delta_range[1] - delta_range[0]) / sampling_precision)),
#         )
#
#     sigma = np.sqrt(np.divide(2, np.pi)) * mu
#     K_max = int(np.ceil(100.0 * sigma / min_possible_exec_time))
#     p_matrix = np.empty(shape=(Ts_values.size, delta_values.size), dtype=float)
#
#     with Pool() as pool:
#         futures = deque()
#         for (Ti, Ts), (di, delta) in itertools.product(
#             enumerate(Ts_values),
#             enumerate(delta_values),
#         ):
#             fut = pool.apply_async(
#                 _energy_penalty_with_offset,
#                 kwds=dict(
#                     Ts=Ts,
#                     delta=delta,
#                     alpha=alpha,
#                     beta=beta,
#                     sigma=sigma,
#                     K_max=K_max,
#                 ),
#                 callback=lambda x: p_matrix.__setitem__((Ti, di), x),
#             )
#             futures.append(fut)
#
#         for fut in futures:
#             fut.get()


class BaseAdaptiveFrameSamplingModel(BaseFrameSamplingModel, metaclass=abc.ABCMeta):
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

    def __init__(
        self,
        probabilities: pd.DataFrame,
        execution_time_model: ExecutionTimeModel,
        success_tag: str = "success",
        init_network_time_guess_seconds: float = 0.3,  # based on exp data
        processing_time_seconds: float = 0.0,  # 0.3,  # taken from experimental data
        # idle_factor: float = 4.0,
        # busy_factor: float = 6.0,  # TODO: document, based on power consumption
        step_delay_cost_window: int = 5,
    ):
        """

        Parameters
        ----------
        probabilities
            Frame probabilities
        execution_time_model
            An execution time model to predict the next steps execution time.
        success_tag

        init_network_time_guess_seconds
            Initial guess for the network time.
        processing_time_seconds
            Factor, expressed in seconds, that is subtracted from frame round-trip
            times, representing the time processing took on the backend.
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
        self._delay_costs = deque(
            [init_network_time_guess_seconds],
            maxlen=step_delay_cost_window,
        )

        # self._idle_factor = idle_factor
        # self._busy_factor = busy_factor

        self._processing_time = processing_time_seconds

    def step_iterator(
        self,
        target_time: float,
        ttf: float,
        # infinite: bool = False,
    ) -> Iterator[Tuple[str, float]]:

        step_start = time.monotonic()
        # step_rtts = deque()

        # Tc = (
        #     np.mean(self._network_times)
        #     if len(self._network_times) > 0
        #     else self._initial_nt_guess
        # )
        self._timing_model.set_ttf(ttf)

        beta = 2.0  # 1.0, trying something,
        alpha = float(np.mean(self._delay_costs))

        for i, target_instant in enumerate(
            _aperiodic_instant_iterator(
                mu=self._timing_model.get_expected_execution_time(),
                alpha=alpha,
                beta=beta,
                # alpha=self._current_rtt_mean *
                # (self._busy_factor - self._idle_factor),
                # beta=self._idle_factor,
            )
        ):
            time.sleep(max(0.0, target_instant - (time.monotonic() - step_start)))
            instant = (tsend := time.monotonic()) - step_start
            yield self.get_frame_at_instant(instant, target_time), instant
            dt = time.monotonic() - tsend

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
