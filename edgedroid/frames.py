from __future__ import annotations

import tarfile
from collections import deque
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Sequence

import cv2
import nptyping as npt
import parse
import yaml


class FrameSet:
    def __init__(self,
                 name: str,
                 initial_frame: npt.NDArray,
                 steps: Sequence[Dict[str, npt.NDArray]]):
        self._name = name
        self._init_frame = initial_frame
        self._steps = tuple(steps)
        self._num_steps = len(self._steps)

    def __str__(self) -> str:
        return yaml.safe_dump(
            {
                'name'         : self._name,
                'num_steps'    : self.step_count,
                'initial_frame': f'{len(self._init_frame.tobytes())} bytes',
                'steps'        : [
                    {
                        tag: f'{len(data.tobytes())} bytes'
                        for tag, data in step.items()
                    } for step in self._steps
                ]
            }
        )

    @property
    def step_count(self) -> int:
        return self._num_steps

    @property
    def name(self) -> str:
        return self._name

    def get_initial_frame(self) -> npt.NDArray:
        return self._init_frame.copy()

    def get_frame(self,
                  step_index: Any,
                  frame_tag: str) -> npt.NDArray:
        return self._steps[step_index][frame_tag].copy()

    @classmethod
    def from_datafile(cls, tarfile_path: PathLike) -> FrameSet:
        """
        Opens a frame datafile and parses it.

        TODO: Add specification on frame datafiles.

        Parameters
        ----------
        tarfile_path
            Path to the datafile.

        Returns
        -------

        """

        tarfile_path = Path(tarfile_path).resolve()
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir).resolve()
            with tarfile.open(tarfile_path, 'r:*') as tfile:
                tfile.extractall(tmpdir)

            # load metadata
            with (tmpdir / 'metadata.yml').open('r') as fp:
                metadata = yaml.safe_load(fp)

            # load initial frame
            init_frame = cv2.imread(str(tmpdir / 'initial.jpeg'),
                                    flags=cv2.IMREAD_UNCHANGED)

            # load frames
            steps = deque()
            for step_i in range(metadata['num_steps']):
                # open corresponding directory
                step_dir = tmpdir / f'step_{step_i:02d}'
                step_dict = {}

                # iterate over the files
                for frame_img in step_dir.glob('*.jpeg'):
                    parse_res = parse.parse('{tag}.jpeg', frame_img.name)
                    img_data = cv2.imread(str(frame_img),
                                          flags=cv2.IMREAD_UNCHANGED)
                    step_dict[parse_res['tag']] = img_data

                steps.append(step_dict)

        return FrameSet(
            name=metadata['task_name'],
            initial_frame=init_frame,
            steps=steps
        )
