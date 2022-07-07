from __future__ import annotations

import abc
import itertools
from collections import deque

import numpy as np
import pandas as pd
from numpy import typing as npt

from .base import BaseFrameSamplingModel
from ..timings import ExecutionTimeModel


def _sampling_instants(
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
