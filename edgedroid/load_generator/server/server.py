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
import queue
import socket
import threading
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
) -> pd.DataFrame:
    logger.info(f"Starting LEGO task '{task_name}'")
    records = deque()

    task = LEGOTask(e_data.load_default_task(task_name))

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


class WritingThread(threading.Thread, contextlib.AbstractContextManager):
    def __init__(self, output_path: pathlib.Path):
        super().__init__()
        self._output = output_path.resolve()
        self._in_q = queue.Queue()
        self._running = threading.Event()

    def push_records(self, records: pd.DataFrame) -> None:
        self._in_q.put_nowait(records)

    def stop(self) -> None:
        logger.debug("Stopping file writing thread")
        self._in_q.join()
        self._running.clear()
        self.join()
        logger.debug("File writing thread stopped")

    def __enter__(self) -> WritingThread:
        self.start()
        super(WritingThread, self).__enter__()
        return self

    def start(self) -> None:
        logger.debug("Starting file writing thread")
        self._running.set()
        super(WritingThread, self).start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
        return super(WritingThread, self).__exit__(exc_type, exc_val, exc_tb)

    def run(self) -> None:
        self._running.set()
        with self._output.open("w+") as fp:
            while self._running.is_set():
                try:
                    records = self._in_q.get(timeout=0.1)

                    # got records
                    fp.seek(0)

                    try:
                        data = pd.concat((pd.read_csv(fp), records), ignore_index=False)
                    except pd.errors.EmptyDataError:
                        # empty file, guess we have just begun writing
                        data = records

                    fp.seek(0)
                    data.to_csv(fp, index=True, header=True)
                    fp.flush()
                    del data

                    self._in_q.task_done()
                except queue.Empty:
                    continue


def serve_LEGO_task(
    task: str,
    port: int,
    output_path: pathlib.Path,
    bind_address: str = "0.0.0.0",
    one_shot: bool = True,
) -> None:
    with contextlib.ExitStack() as stack:
        # enter context
        server_sock: socket.SocketType = stack.enter_context(
            socket.create_server(
                (bind_address, port), family=socket.AF_INET, backlog=1, reuse_port=True
            )
        )
        wt: WritingThread = stack.enter_context(WritingThread(output_path=output_path))

        logger.info(f"Serving LEGO task {task} on {bind_address}:{port}/tcp")
        logger.debug(f"One-shot mode: {'on' if one_shot else 'off'}")
        try:
            clients_served = 0
            while True:
                with accept_context(server_sock) as (conn, _):
                    clients_served += 1
                    results = server(task, conn)
                    results["client"] = clients_served
                    results = results.reset_index().set_index(
                        ["client", "seq"], verify_integrity=True
                    )
                    wt.push_records(results)
                if one_shot:
                    logger.warning("One-shot server, shutting down")
                    break
        except KeyboardInterrupt:
            logger.warning("Got keyboard interrupt")
            logger.warning("Shutting down!")
