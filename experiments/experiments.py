from typing import Any, Callable, Dict, NamedTuple

import scipy.stats as stats

from edgedroid.models import *


class ExperimentConfig(NamedTuple):
    timing_model: ExecutionTimeModel
    sampling_scheme: BaseFrameSamplingModel
    metadata: Dict[str, Any] = {}


experiments: Dict[str, Callable[[], ExperimentConfig]] = {
    "empirical-high-adaptive-power-empirical": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=EmpiricalExecutionTimeModel(neuroticism=None)
        ),
        metadata={
            "timing_model": "empirical-high",
            "sampling_scheme": "adaptive-power-empirical",
        },
    ),
    # ---
    "empirical-high-adaptive-power-empirical-low": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=EmpiricalExecutionTimeModel(neuroticism=0.0)
        ),
        metadata={
            "timing_model": "empirical-high",
            "sampling_scheme": "adaptive-power-empirical-low",
        },
    ),
    # ---
    "empirical-high-adaptive-power-empirical-high": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=EmpiricalExecutionTimeModel(neuroticism=1.0)
        ),
        metadata={
            "timing_model": "empirical-high",
            "sampling_scheme": "adaptive-power-empirical-high",
        },
    ),
    # ---
    "empirical-high-adaptive-power-theoretical-exgaussian": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=TheoreticalExecutionTimeModel(
                neuroticism=None, distribution=stats.exponnorm
            )
        ),
        metadata={
            "timing_model": "empirical-high",
            "sampling_scheme": "adaptive-power-theoretical-exgaussian",
        },
    ),
    # ---
    "empirical-high-adaptive-power-theoretical-exgaussian-low": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=TheoreticalExecutionTimeModel(
                neuroticism=0.0, distribution=stats.exponnorm
            )
        ),
        metadata={
            "timing_model": "empirical-high",
            "sampling_scheme": "adaptive-power-theoretical-exgaussian-low",
        },
    ),
    # ---
    "empirical-high-adaptive-power-theoretical-exgaussian-high": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=TheoreticalExecutionTimeModel(
                neuroticism=1.0, distribution=stats.exponnorm
            )
        ),
        metadata={
            "timing_model": "empirical-high",
            "sampling_scheme": "adaptive-power-theoretical-exgaussian-high",
        },
    ),
    "empirical-high-adaptive-power-fitted-naive-exgaussian": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=FittedNaiveExecutionTimeModel(
                dist=stats.exponnorm,
            )
        ),
        metadata={
            "timing_model": "empirical-high",
            "sampling_scheme": "adaptive-power-fitted-naive-exgaussian",
        },
    ),
    "empirical-high-greedy": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel(neuroticism=1.0),
        sampling_scheme=ZeroWaitFrameSamplingModel.from_default_data(),
        metadata={"timing_model": "empirical-high", "sampling_scheme": "greedy"},
    ),
    "empirical-low-greedy": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel(neuroticism=0.0),
        sampling_scheme=ZeroWaitFrameSamplingModel.from_default_data(),
        metadata={"timing_model": "empirical-low", "sampling_scheme": "greedy"},
    ),
    "theoretical-exgaussian-high-greedy": lambda: ExperimentConfig(
        timing_model=TheoreticalExecutionTimeModel(
            neuroticism=1.0, distribution=stats.exponnorm
        ),
        sampling_scheme=ZeroWaitFrameSamplingModel.from_default_data(),
        metadata={
            "timing_model": "theoretical-exgaussian-high",
            "sampling_scheme": "greedy",
        },
    ),
    "theoretical-exgaussian-low-greedy": lambda: ExperimentConfig(
        timing_model=TheoreticalExecutionTimeModel(
            neuroticism=0.0, distribution=stats.exponnorm
        ),
        sampling_scheme=ZeroWaitFrameSamplingModel.from_default_data(),
        metadata={
            "timing_model": "theoretical-exgaussian-low",
            "sampling_scheme": "greedy",
        },
    ),
    "fitted-naive-exgaussian-greedy": lambda: ExperimentConfig(
        timing_model=FittedNaiveExecutionTimeModel(dist=stats.exponnorm),
        sampling_scheme=ZeroWaitFrameSamplingModel.from_default_data(),
        metadata={
            "timing_model": "fitted-naive-exgaussian",
            "sampling_scheme": "greedy",
        },
    ),
}
__all__ = ["experiments"]
