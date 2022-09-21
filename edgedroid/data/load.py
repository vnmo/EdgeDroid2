#  Copyright (c) 2022 Manuel Olguín Muñoz <molguin@kth.se>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from importlib import resources
from importlib.resources import as_file
from typing import List, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pooch
from loguru import logger
from pandas import arrays

__all__ = [
    "load_default_frame_probabilities",
    "load_default_exec_time_data",
    "load_default_trace",
    "load_default_task",
]

from ..models import FrameSet

# _default_neuro_bins = np.array([-np.inf, 1 / 3, 2 / 3, np.inf])
# _default_impairment_bins = np.array([-np.inf, 1.0, 2.0, np.inf])
# _default_duration_bins = np.array([0, 5, 10, np.inf])

_default_traces = {
    "square00": "md5:29b13222390bc5587e2306c41fc8ba01",
    "square01": "md5:56aceb607e201ae7a7fea28b933567c1",
    "square03": "md5:0cada8d6a76ffaf58da39a287bfb7373",
    "square02": "md5:43f77a7466c9a0dbc1773b84f2f02512",
    "square06": "md5:64cd191c24f5f06d75c55917666241bd",
    "square07": "md5:bd99adbfc911add8423e7b362bc035c2",
    "square11": "md5:e3240d7955ce0869e0de184c4f364f86",
    "square05": "md5:3ceb7c63572b6938e16e68ef49e977bc",
    "square04": "md5:6df051d4c260980e63281dbef090b31a",
    "square10": "md5:935800f3a284020c64aef988d72f82da",
    "square09": "md5:bcb586ae17cc4b96829b4ac0e6f0067b",
    "square08": "md5:f7f4aea7235545bbe10a087c5ec2c4d3",
    "test": "md5:50fd4887b4d4a6312befb9758ab31fd0",
}


class ExecTimeDataParams(NamedTuple):
    base_data: pd.DataFrame
    neuroticism_bins: arrays.IntervalArray
    impairment_bins: arrays.IntervalArray
    duration_bins: arrays.IntervalArray


def load_default_exec_time_data() -> ExecTimeDataParams:
    from . import resources as edgedroid_resources

    data_file = resources.files(edgedroid_resources).joinpath(
        "model_exec_times.parquet"
    )

    bins_file = resources.files(edgedroid_resources).joinpath("default_bins.npz")

    bins_map = np.load(bins_file)

    with as_file(data_file) as fp:
        return ExecTimeDataParams(
            pd.read_parquet(fp),
            arrays.IntervalArray.from_breaks(bins_map["neuroticism"], closed="left"),
            arrays.IntervalArray.from_breaks(bins_map["impairment"], closed="left"),
            arrays.IntervalArray.from_breaks(bins_map["duration"], closed="left"),
        )


def load_default_frame_probabilities() -> pd.DataFrame:
    from . import resources as edgedroid_resources

    frame_prob_file = resources.files(edgedroid_resources).joinpath(
        "frame_probabilities.csv"
    )
    with as_file(frame_prob_file) as fp:
        return pd.read_csv(fp)


def load_default_trace(trace_name: str, truncate: int = -1) -> FrameSet:
    logger.debug(
        f"Loading default trace '{trace_name}' "
        f"with known hash {_default_traces[trace_name]}"
    )

    trace_url = (
        f"https://github.com/molguin92/EdgeDroid2/releases/download"
        f"/v1.0.0-traces/{trace_name}.npz"
    )
    logger.debug(f"Attempting to fetch trace from {trace_url} if necessary")

    trace_path = pooch.retrieve(
        url=trace_url, known_hash=_default_traces[trace_name], progressbar=True
    )

    return FrameSet.from_datafile(
        task_name=trace_name,
        trace_path=trace_path,
        truncate=truncate,
    )


class TaskLoadException(Exception):
    pass


def load_default_task(
    task_name: str,
    truncate: int = -1,
) -> List[npt.NDArray[int]]:
    from . import resources as edgedroid_resources

    state_file = resources.files(edgedroid_resources).joinpath(f"{task_name}.npz")
    with as_file(state_file) as fp:
        states = np.load(str(fp))

    num_states = len(states)
    if truncate >= 0:
        if len(states) < truncate:
            raise TaskLoadException(
                f"Task has {num_states} steps, which is less than the desired "
                f"truncated length of {truncate} steps."
            )

        num_states = truncate

    return [states[str(i)] for i in range(num_states)]
