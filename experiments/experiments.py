from typing import Callable, Dict, NamedTuple

import scipy.stats as stats

from edgedroid.models import *


class ExperimentConfig(NamedTuple):
    timing_model: ExecutionTimeModel
    sampling_scheme: BaseFrameSamplingModel


experiments: Dict[str, Callable[[], ExperimentConfig]] = {
    "empirical-high-adaptive-power-empirical": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel.from_default_data(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=EmpiricalExecutionTimeModel.from_default_data(
                neuroticism=None
            )
        ),
    ),
    "empirical-high-adaptive-power-theoretical-exgaussian": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel.from_default_data(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=TheoreticalExecutionTimeModel.from_default_data(
                neuroticism=None,
                distribution=stats.exponnorm,
            )
        ),
    ),
    "empirical-high-adaptive-power-theoretical-rayleigh": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel.from_default_data(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=TheoreticalExecutionTimeModel.from_default_data(
                neuroticism=None,
                distribution=stats.rayleigh,
            )
        ),
    ),
    "empirical-high-adaptive-power-fitted-naive-exgaussian": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel.from_default_data(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=FittedNaiveExecutionTimeModel.from_default_data(
                distribution=stats.exponnorm,
            )
        ),
    ),
    "empirical-high-adaptive-power-fitted-naive-rayleigh": lambda: ExperimentConfig(
        timing_model=EmpiricalExecutionTimeModel.from_default_data(neuroticism=1.0),
        sampling_scheme=AperiodicPowerFrameSamplingModel.from_default_data(
            execution_time_model=FittedNaiveExecutionTimeModel.from_default_data(
                distribution=stats.rayleigh,
            )
        ),
    ),
}
__all__ = ["experiments"]
