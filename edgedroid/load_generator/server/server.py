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
from __future__ import annotations

import contextlib
import pathlib
import socket
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterator, Tuple

import pandas as pd
from gabriel_lego import FrameResult, LEGOTask
from loguru import logger

from ..common import frame_stream_unpack, pack_response
from ... import data as e_data


@dataclass(frozen=True, eq=True)
class FrameRecord:
    seq: int
    received: float
    received_monotonic: float
    processed: float
    processed_monotonic: float
    processing_time: float
    result: FrameResult

    def to_dict(self) -> Dict[str, int | float | FrameResult]:
        return asdict(self)


def server(
    task_name: str,
    sock: socket.SocketType,
    result_cb: Callable[[FrameResult], None] = lambda _: None,
    truncate: int = -1,
) -> pd.DataFrame:
    logger.info(f"Starting LEGO task trace '{task_name}'")
    records = deque()

    task = LEGOTask(e_data.load_default_task(task_name, truncate=truncate))

    with contextlib.closing(frame_stream_unpack(sock)) as frame_stream:
        for seq, image_data in frame_stream:
            recv_time_mono = time.monotonic()
            recv_time = time.time()

            logger.info(f"Received frame with SEQ {seq}")
            result = task.submit_frame(image_data)

            proc_time_mono = time.monotonic()
            proc_time = time.time()

            logger.debug(f"Processing result: {result}")
            result_cb(result)

            if result == FrameResult.SUCCESS:
                logger.success(
                    f"Frame with SEQ {seq} triggers advancement to next step"
                )
                transition = True
            else:
                transition = False

            sock.sendall(
                pack_response(
                    transition,
                    task.get_current_guide_illustration(),
                    task.get_current_instruction(),
                )
            )

            # finally, store frame record
            records.append(
                FrameRecord(
                    seq=seq,
                    result=result,
                    received=recv_time,
                    received_monotonic=recv_time_mono,
                    processed=proc_time,
                    processed_monotonic=proc_time_mono,
                    processing_time=proc_time_mono - recv_time_mono,
                )
            )

    # finally, return the recorded frames
    return pd.DataFrame(records).set_index("seq")


@contextlib.contextmanager
def accept_context(
    sock: socket.SocketType,
) -> Iterator[Tuple[socket.SocketType, Tuple[str, int]]]:
    logger.info("Waiting for connections...")
    conn, (peer_addr, peer_port) = sock.accept()
    logger.info(f"Opened connection to {peer_addr}:{peer_port}")
    try:
        yield conn, (peer_addr, peer_port)
    finally:
        logger.warning(f"Closing connection to {peer_addr}:{peer_port}")
        conn.close()


def serve_LEGO_task(
    task_name: str,
    port: int,
    output_path: pathlib.Path,
    bind_address: str = "0.0.0.0",
    truncate: int = -1,
) -> None:
    with contextlib.ExitStack() as stack:
        # enter context
        server_sock: socket.SocketType = stack.enter_context(
            socket.create_server(
                (bind_address, port), family=socket.AF_INET, backlog=1, reuse_port=True
            )
        )

        logger.info(f"Serving LEGO task {task_name} on {bind_address}:{port}/tcp")
        if truncate >= 0:
            logger.info(f"Task is truncated to {truncate} steps")
        # logger.debug(f"One-shot mode: {'on' if one_shot else 'off'}")
        try:
            with accept_context(server_sock) as (conn, _):
                server(task_name, conn, truncate=truncate).to_csv(output_path)
        except KeyboardInterrupt:
            logger.warning("Got keyboard interrupt, aborting")
            logger.warning("Shutting down!")
