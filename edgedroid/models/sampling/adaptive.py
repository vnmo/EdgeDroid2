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

    sigma = np.sqrt(np.divide(2, np.pi)) * mu
    const_factor = np.float_power(
        3 * sigma * np.sqrt(np.divide(alpha, 2 * beta)),
        np.divide(2, 3),
    )

    t0 = const_factor * np.float_power(0, np.divide(2, 3))
    t1 = const_factor * np.float_power(1, np.divide(2, 3))
    instants = deque([t0, t1])

    prev_dt = t1 - t0

    for n in itertools.count(start=2, step=1):
        tn = const_factor * np.float_power(n, np.divide(2, 3))

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
    def __init__(
        self,
        probabilities: pd.DataFrame,
        execution_time_model: ExecutionTimeModel,
        initial_network_time_guess: float,
        processing_time: float,
        success_tag: str = "success",
        idle_factor: float = 4.0,
        busy_factor: float = 6.0,  # TODO: document, based on power consumption
    ):
        super(AperiodicFrameSamplingModel, self).__init__(
            probabilities=probabilities,
            execution_time_model=execution_time_model,
            success_tag=success_tag,
        )

        self._initial_nt_guess = initial_network_time_guess
        self._network_time_sum = 0.0
        self._network_time_count = 0

        self._idle_factor = idle_factor
        self._busy_factor = busy_factor

        self._processing_time = processing_time

    def step_iterator(
        self,
        target_time: float,
        delay: float,
        infinite: bool = False,
    ) -> Iterator[Tuple[str, float]]:

        step_start = time.monotonic()

        Tc = (
            self._network_time_sum / self._network_time_count
            if self._network_time_count > 0
            else self._initial_nt_guess
        )
        self._timing_model.set_delay(delay)

        sampling_instants = _aperiodic_sampling_instants(
            mu=self._timing_model.get_expected_execution_time(),
            alpha=Tc * (self._busy_factor - self._idle_factor),
            beta=self._idle_factor,
        )

        for instant in itertools.chain(
            sampling_instants[:1],
            itertools.repeat(sampling_instants[-1]),  # repeat the last instant forever.
            # In practice, this makes the sampling fall back to zero-wait.
        ):
            time.sleep(max(0, instant - (time.monotonic() - step_start)))
            tsend = time.monotonic()
            yield self.get_frame_at_instant(instant, target_time), instant
            dt = time.monotonic() - tsend

            self._network_time_sum += dt - self._processing_time
            self._network_time_count += 1

            if instant > target_time and not infinite:
                return
