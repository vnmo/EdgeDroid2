from importlib import resources
from importlib.resources import as_file
from typing import List, Tuple

import numpy as np
import pandas as pd
import pooch
from pandas import arrays
import numpy.typing as npt

__all__ = [
    "load_default_frame_probabilities",
    "load_default_exec_time_data",
    "load_default_trace",
    "load_default_task",
]

from ..models import FrameSet

_default_neuro_bins = np.array([-np.inf, 1 / 3, 2 / 3, np.inf])
_default_impairment_bins = np.array([-np.inf, 1.0, 2.0, np.inf])
_default_duration_bins = np.array([0, 5, 10, np.inf])

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
}


def load_default_exec_time_data() -> Tuple[
    pd.DataFrame, arrays.IntervalArray, arrays.IntervalArray, arrays.IntervalArray
]:
    from . import resources as edgedroid_resources

    data_file = resources.files(edgedroid_resources).joinpath(
        "model_exec_times.parquet"
    )

    with as_file(data_file) as fp:
        return (
            pd.read_parquet(fp).reset_index(),
            arrays.IntervalArray.from_breaks(_default_neuro_bins, closed="left"),
            arrays.IntervalArray.from_breaks(_default_impairment_bins, closed="left"),
            arrays.IntervalArray.from_breaks(_default_duration_bins, closed="left"),
        )


def load_default_frame_probabilities() -> pd.DataFrame:
    from . import resources as edgedroid_resources

    frame_prob_file = resources.files(edgedroid_resources).joinpath(
        "frame_probabilities.csv"
    )
    with as_file(frame_prob_file) as fp:
        return pd.read_csv(fp)


def load_default_trace(trace_name: str) -> FrameSet:
    trace_url = (
        f"https://github.com/molguin92/EdgeDroid2/releases/download"
        f"/v1.0.0-traces/{trace_name}.npz"
    )
    trace_path = pooch.retrieve(
        url=trace_url, known_hash=_default_traces[trace_name], progressbar=True
    )

    return FrameSet.from_datafile(task_name=trace_name, trace_path=trace_path)


def load_default_task(task_name: str) -> List[npt.NDArray[int]]:
    from . import resources as edgedroid_resources

    state_file = resources.files(edgedroid_resources).joinpath(f"{task_name}.npz")
    with as_file(state_file) as fp:
        states = np.load(str(fp))

    return [states[str(i)] for i in range(len(states))]
