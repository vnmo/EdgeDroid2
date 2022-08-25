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
from collections import deque
from multiprocessing import Condition, Event, Pipe, Process
from multiprocessing.connection import Connection
from threading import Event as TEvent, Thread
from typing import Deque

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from edgedroid.load_generator.client.client import StreamSocketEmulation
from edgedroid.load_generator.server.server import server
from edgedroid.models import ModelFrame


class ServerProcess(Process):
    def __init__(self, unix_socket_addr: str, task_name: str = "test"):
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
        # enable logging
        logger.enable("edgedroid")

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


class ClientProcess(Process):
    def __init__(
        self,
        task_name: str = "test",
        neuroticism: float = 0.5,
        fade_distance: int = 8,
    ):
        super(ClientProcess, self).__init__()

        self._task = task_name
        self._neuro = neuroticism
        self._fade_dist = fade_distance
        self._is_finished = Event()

        self._frame_recv, self._frame_send = Pipe(duplex=False)
        self._guidance_recv, self._guidance_send = Pipe(duplex=False)

    @property
    def frame_recv_pipe(self) -> Connection:
        return self._frame_recv

    @property
    def guidance_recv_pipe(self) -> Connection:
        return self._guidance_recv

    def is_finished(self) -> bool:
        return self._is_finished.is_set()

    def _frame_callback(self, frame: ModelFrame) -> None:
        self._frame_send.send((frame.frame_data.tolist(), frame.frame_tag))

    def _guide_callback(
        self, transition: bool, img: npt.NDArray, instruction: str
    ) -> None:
        if transition:
            self._guidance_send.send((img.tolist(), instruction))

    def run(self) -> None:
        # enable logging
        logger.enable("edgedroid")

        self._is_finished.clear()
        emulation = StreamSocketEmulation(
            neuroticism=self._neuro,
            trace=self._task,
            fade_distance=self._fade_dist,
            sampling="adaptive-aperiodic",
        )
        socket_addr = f"/tmp/{uuid.uuid4()}.sock"

        logger.info("Starting server process")
        server_proc = ServerProcess(socket_addr, task_name=self._task)

        try:
            server_proc.start()
            server_proc.wait_until_ready()

            # connect the client socket
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                logger.info(f"Connecting to server socket on {socket_addr}")
                sock.connect(socket_addr)
                logger.success("Connection success, starting emulation")
                emulation.emulate(
                    sock,
                    emit_cb=self._frame_callback,
                    resp_cb=self._guide_callback,
                )
                logger.success("Emulation done!")
                print(emulation.get_frame_metrics())
            finally:
                sock.close()
        finally:
            server_proc.join(timeout=1.0)
            self._is_finished.set()


def pipe2deque(pipe: Connection, dq: Deque[npt.NDArray, str], shutdown: TEvent) -> None:
    while not shutdown.is_set():
        if pipe.poll(timeout=0.01):
            img, msg = pipe.recv()
            dq.append((np.array(img, dtype=np.uint8), msg))


def demo_loop(
    task_name: str = "test",
    view_width: int = 1200,
    view_height: int = 300,
):
    # enable logging
    logger.enable("edgedroid")
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        enqueue=True,
        colorize=True,
        backtrace=True,
        diagnose=True,
        catch=True,
    )

    # initial black images
    current_guidance = np.zeros((view_height, view_width // 2, 3), dtype=np.uint8)
    current_input = np.zeros((view_height, view_width // 2, 3), dtype=np.uint8)

    def resize_add_text(img: npt.NDArray, text: str) -> npt.NDArray:
        w = view_width // 2

        img = cv2.resize(img, (w, view_height), cv2.INTER_AREA)
        cv2.putText(
            img=img,
            text=f"{text[:50]}",
            color=(0, 0, 255),
            org=(view_height // 10, w // 10),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            thickness=1,
        )
        return img

    client_proc = ClientProcess(task_name=task_name, neuroticism=0.5, fade_distance=8)

    frame_dq: Deque[npt.NDArray, str] = deque(maxlen=1)
    guide_dq: Deque[npt.NDArray, str] = deque(maxlen=1)
    dq_thread_shutdown = TEvent()

    frame_recv_t = Thread(
        target=pipe2deque,
        args=(
            client_proc.frame_recv_pipe,
            frame_dq,
            dq_thread_shutdown,
        ),
    )
    guide_recv_t = Thread(
        target=pipe2deque,
        args=(
            client_proc.guidance_recv_pipe,
            guide_dq,
            dq_thread_shutdown,
        ),
    )

    # ui loop
    _ = cv2.namedWindow("DEMO")
    try:
        frame_recv_t.start()
        guide_recv_t.start()
        client_proc.start()

        while not client_proc.is_finished():
            try:
                guidance, msg = guide_dq.popleft()
                current_guidance = resize_add_text(guidance, msg)
            except IndexError:
                pass

            try:
                frame, tag = frame_dq.popleft()
                current_input = resize_add_text(frame, tag)
            except IndexError:
                pass

            output_img = np.concatenate((current_guidance, current_input), axis=1)
            cv2.imshow("DEMO", output_img)
            cv2.waitKey(25)
    finally:
        client_proc.join()
        dq_thread_shutdown.set()
        frame_recv_t.join()
        guide_recv_t.join()


if __name__ == "__main__":
    demo_loop()
