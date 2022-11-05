import abc
from typing import NamedTuple

import numpy as np

from edgedroid.models import ExecutionTimeModel
from edgedroid.models.sampling.adaptive import _aperiodic_instant_iterator


class StepResults(NamedTuple):
    execution_time: float
    duration: float
    rtt: float
    ttf: float
    wait_time: float
    num_samples: float
    energy: float
    P0: float
    Pc: float
    t0: float
    tc: float


class SamplingResults(NamedTuple):
    last_sampling_instant: float
    num_samples: int


class ConstantRTTSampler(abc.ABC):
    def __init__(self, t_net: float, t_proc: float, P0: float, Pc: float):
        self._rtt = t_net + t_proc
        self._tc = t_net
        self._P0 = P0
        self._Pc = Pc

    @abc.abstractmethod
    def reset(self):
        pass

    def calculate_energy(
            self,
            duration: float,
            num_samples: int,
    ) -> float:
        comm_time = self._tc * num_samples
        idle_time = duration - comm_time
        return (comm_time * self._Pc) + (idle_time * self._P0)

    @abc.abstractmethod
    def _sample_step(self, prev_ttf: float, execution_time: float) -> SamplingResults:
        pass

    def sample_step(self, prev_ttf: float, execution_time: float) -> StepResults:
        last_sample_instant, num_samples = self._sample_step(prev_ttf, execution_time)
        duration = last_sample_instant + self._rtt
        ttf = duration - execution_time
        wait = ttf - self._rtt
        energy = self.calculate_energy(duration, num_samples)

        return StepResults(
            execution_time=execution_time,
            duration=duration,
            rtt=self._rtt,
            ttf=ttf,
            wait_time=wait,
            energy=energy,
            num_samples=num_samples,
            P0=self._P0,
            Pc=self._Pc,
            t0=self._rtt - self._tc,
            tc=self._tc,
        )


class IdealConstantRTTSampler(ConstantRTTSampler):
    def _sample_step(self, prev_ttf: float, execution_time: float) -> SamplingResults:
        return SamplingResults(execution_time, 1)

    def reset(self):
        pass


class GreedyConstantRTTSampler(ConstantRTTSampler):
    def _sample_step(self, prev_ttf: float, execution_time: float) -> SamplingResults:
        instant = 0.0
        samples = 1

        while instant <= execution_time:
            instant += self._rtt
            samples += 1

        return SamplingResults(instant, samples)

    def reset(self):
        pass


class JunjuesConstantRTTSampler(ConstantRTTSampler):
    def __init__(
            self,
            cdf_estimator: ExecutionTimeModel,
            min_sr: float,
            alpha: float,
            t_net: float,
            t_proc: float,
            P0: float,
            Pc: float,
    ):
        super(JunjuesConstantRTTSampler, self).__init__(t_net, t_proc, P0, Pc)
        self._estimator = cdf_estimator
        self._estimator.reset()
        self._sr_min = min_sr
        self._sr_max = 1 / self._rtt
        self._sr_diff = self._sr_max - self._sr_min
        self._alpha = alpha

    def reset(self):
        self._estimator.reset()

    def _sample_step(self, prev_ttf: float, execution_time: float) -> SamplingResults:
        self._estimator.advance(prev_ttf)
        instant = 0.0
        samples = 0

        while instant <= execution_time:
            sr = min(
                self._sr_min
                + (
                        self._alpha
                        * self._sr_diff
                        * self._estimator.get_cdf_at_instant(instant)
                ),
                self._sr_max,
            )

            instant += 1 / sr
            samples += 1

        return SamplingResults(instant, samples)


class BaseOptimumSampler(ConstantRTTSampler, metaclass=abc.ABCMeta):
    def __init__(
            self,
            estimator: ExecutionTimeModel,
            t_net: float,
            t_proc: float,
            P0: float,
            Pc: float,
    ):
        super(BaseOptimumSampler, self).__init__(t_net, t_proc, P0, Pc)
        self._estimator = estimator
        self._estimator.reset()

    def reset(self):
        self._estimator.reset()

    @abc.abstractmethod
    def get_alpha(self) -> float:
        pass

    @abc.abstractmethod
    def get_beta(self) -> float:
        pass

    def _sample_step(self, prev_ttf: float, execution_time: float) -> SamplingResults:
        self._estimator.advance(prev_ttf)

        instant_iter = _aperiodic_instant_iterator(
            mu=self._estimator.get_mean_execution_time(),
            alpha=self.get_alpha(),
            beta=self.get_beta(),
        )

        instant = 0.0
        samples = 0

        while instant <= execution_time:
            instant = max(next(instant_iter), instant + self._rtt)
            samples += 1

        return SamplingResults(instant, samples)


class SamplingOptimumSampler(BaseOptimumSampler):
    def __init__(
            self,
            estimator: ExecutionTimeModel,
            max_wait: float,
            t_net: float,
            t_proc: float,
            P0: float,
            Pc: float,
            beta: float = 1.0,
    ):
        super(SamplingOptimumSampler, self).__init__(estimator, t_net, t_proc, P0, Pc)

        self._w0 = max_wait
        self._beta = beta

    def get_alpha(self) -> float:
        mu = self._estimator.get_mean_execution_time()
        sigma = np.sqrt(np.divide(2, np.pi)) * mu
        return 1.9 * (self._w0 / sigma) * self.get_beta()

    def get_beta(self) -> float:
        return self._beta


class EnergyOptimumSampler(BaseOptimumSampler):
    def get_alpha(self) -> float:
        return self._tc * (self._Pc - self._P0)

    def get_beta(self) -> float:
        return self._P0
