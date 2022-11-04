import abc
from typing import Callable, NamedTuple

from edgedroid.models.sampling.adaptive import _aperiodic_instant_iterator


class SamplingResult(NamedTuple):
    duration: float
    num_samples: int


class Sampler(Callable[[float, float], SamplingResult]):
    @abc.abstractmethod
    def __call__(self, exec_time: float, rtt: float) -> SamplingResult:
        pass


class GreedySampler(Sampler):
    def __call__(self, exec_time: float, rtt: float) -> SamplingResult:
        instant = 0.0
        samples = 1

        while instant <= exec_time:
            samples += 1
            instant += rtt

        return SamplingResult(instant + rtt, samples)


class IdealSampler(Sampler):
    def __call__(self, exec_time: float, rtt: float) -> SamplingResult:
        return SamplingResult(exec_time + rtt, 1)


class JunjuesSampler(Sampler):
    def __init__(self,
                 cdf: Callable[[float], float],
                 min_sr: float,
                 alpha: float,
                 ):
        self._cdf = cdf
        self._sr_min = min_sr
        self._alpha = alpha

    def __call__(self, exec_time: float, rtt: float) -> SamplingResult:
        sr_max = 1 / rtt

        sr_diff = sr_max - self._sr_min

        instant = 0.0
        samples = 0

        while instant <= exec_time:
            sr = min(
                self._sr_min + (self._alpha * sr_diff * self._cdf(instant)),
                sr_max
            )

            instant += (1 / sr)
            samples += 1

        return SamplingResult(instant + rtt, samples)


class OptimumSampler(Sampler):
    def __init__(
            self,
            mean_exec_time_estimator: Callable[[], float],
            alpha_calculator: Callable[[], float],
            beta_calculator: Callable[[], float],
    ):
        self._mu_estimator = mean_exec_time_estimator
        self._alpha_gen = alpha_calculator
        self._beta_gen = beta_calculator

    def __call__(self, exec_time: float, rtt: float) -> SamplingResult:
        instant_iter = _aperiodic_instant_iterator(
            mu=self._mu_estimator(),
            alpha=self._alpha_gen(),
            beta=self._beta_gen()
        )

        instant = next(instant_iter)
        samples = 1

        while instant <= exec_time:
            instant = max(next(instant_iter), instant + rtt)
            samples += 1

        return SamplingResult(instant + rtt, samples)


def calculate_energy(
        P0: float,
        Pc: float,
        tc: float,
        sampling_res: SamplingResult,
) -> float:
    comm_time = tc * sampling_res.num_samples
    idle_time = sampling_res.duration - comm_time
    return (comm_time * Pc) + (idle_time * P0)
