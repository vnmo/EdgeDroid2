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

import socket
import sys
import uuid
from multiprocessing import Condition, Event, Process

from loguru import logger

from edgedroid.load_generator.client.client import StreamSocketEmulation
from edgedroid.load_generator.server.server import server


class ServerProcess(Process):
    def __init__(self, unix_socket_addr: str, task_name: str = "square00"):
        super(ServerProcess, self).__init__()
        self._sock_addr = unix_socket_addr
        self._task = task_name
        self._ready_cond = Condition()
        self._is_ready = Event()

    def wait_until_ready(self) -> None:
        with self._ready_cond:
            while not self._is_ready.is_set():
                self._ready_cond.wait()

    def run(self) -> None:
        # create a unix socket to listen on
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(self._sock_addr)
            sock.listen(1)
            logger.info(f"Server listening on {self._sock_addr}")

            with self._ready_cond:
                self._is_ready.set()
                self._ready_cond.notify_all()

            conn, peer_addr = sock.accept()
            logger.info("Server accepts connection")
            try:
                results = server(self._task, conn)
            finally:
                logger.warning("Server closing connection")
                conn.close()
        finally:
            sock.close()


def demo_loop(trace="test"):
    # enable logging
    logger.enable("edgedroid")
    logger.remove()

    logger.add(
        sys.stderr,
        enqueue=True,
        colorize=True,
        backtrace=True,
        diagnose=True,
        catch=True,
    )

    emulation = StreamSocketEmulation(
        neuroticism=0.5,
        trace=trace,
        fade_distance=8,
    )

    socket_addr = f"/tmp/{uuid.uuid4()}.sock"

    logger.info("Starting server process")

    server_proc = ServerProcess(socket_addr, task_name=trace)

    try:
        server_proc.start()
        server_proc.wait_until_ready()

        # connect the client socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            logger.info(f"Connecting to server socket on {socket_addr}")
            sock.connect(socket_addr)
            logger.success("Connection success, starting emulation")
            emulation.emulate(sock)
            logger.success("Emulation done!")
        finally:
            sock.close()
    finally:
        server_proc.join(timeout=1.0)


if __name__ == "__main__":
    demo_loop()
