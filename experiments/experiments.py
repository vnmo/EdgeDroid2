from typing import Any, Callable, Dict, NamedTuple

import scipy.stats as stats

from edgedroid.models import *


class ExperimentConfig(NamedTuple):
    timing_model: ExecutionTimeModel
    sampling_scheme: BaseFrameSamplingModel
    metadata: Dict[str, Any] = {}


experiments: Dict[str, Callable[[], ExperimentConfig]] = {
    "empirical-high-adaptive-power-empirical": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel.from_default_data(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=EmpiricalExecutionTimeModel.from_default_data(
                neuroticism=None
            )
        ),
        metadata={
            "timing_model": "empirical-high",
            "sampling_scheme": "adaptive-power-empirical",
        },
    ),
    # ---
    "empirical-high-adaptive-power-empirical-low": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel.from_default_data(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=EmpiricalExecutionTimeModel.from_default_data(
                neuroticism=0.0
            )
        ),
        metadata={
            "timing_model": "empirical-high",
            "sampling_scheme": "adaptive-power-empirical-low",
        },
    ),
    # ---
    "empirical-high-adaptive-power-empirical-high": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel.from_default_data(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=EmpiricalExecutionTimeModel.from_default_data(
                neuroticism=1.0
            )
        ),
        metadata={
            "timing_model": "empirical-high",
            "sampling_scheme": "adaptive-power-empirical-high",
        },
    ),
    # ---
    "empirical-high-adaptive-power-theoretical-exgaussian": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel.from_default_data(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=TheoreticalExecutionTimeModel.from_default_data(
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
        timing_model=EmpiricalExecutionTimeModel.from_default_data(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=TheoreticalExecutionTimeModel.from_default_data(
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
        timing_model=EmpiricalExecutionTimeModel.from_default_data(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=TheoreticalExecutionTimeModel.from_default_data(
                neuroticism=1.0, distribution=stats.exponnorm
            )
        ),
        metadata={
            "timing_model": "empirical-high",
            "sampling_scheme": "adaptive-power-theoretical-exgaussian-high",
        },
    ),
    "empirical-high-adaptive-power-fitted-naive-exgaussian": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel.from_default_data(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=FittedNaiveExecutionTimeModel.from_default_data(
                distribution=stats.exponnorm,
            )
        ),
        metadata={
            "timing_model": "empirical-high",
            "sampling_scheme": "adaptive-power-fitted-naive-exgaussian",
        },
    ),
}
__all__ = ["experiments"]
