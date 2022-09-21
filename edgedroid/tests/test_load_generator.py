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
import queue
import socket
import sys
import unittest
from functools import wraps
from threading import Event, Thread
from typing import Any, Callable, Iterator, Optional, Tuple

import numpy as np
from numpy import testing as nptest
from gabriel_lego import FrameResult
from loguru import logger

from ..data import load_default_trace
from ..load_generator import common
from ..load_generator.client.client import StreamSocketEmulation
from ..load_generator.server.server import server
from ..models import ModelFrame


def log_test(fn: Callable) -> Callable:
    @wraps(fn)
    def _wrapper(*args, **kwargs) -> Any:
        logger.info(f"Running {fn}")
        ret = fn(*args, **kwargs)
        logger.success(f"Finished {fn}")
        return ret

    return _wrapper


@contextlib.contextmanager
def client_server_sockets(
    timeout: Optional[float] = None,
) -> Iterator[Tuple[socket.SocketType, socket.SocketType]]:
    logger.info("Opening pair of connected UNIX sockets")
    client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    logger.debug(f"Setting socket timeout values: {timeout:0.3f} seconds")
    client.settimeout(timeout)
    server.settimeout(timeout)
    try:
        yield client, server
    finally:
        logger.info("Closing pair of connected UNIX sockets")
        client.close()
        server.close()


class BytesSocketClient(contextlib.AbstractContextManager, Thread):
    def __init__(self, sock: socket.SocketType) -> None:
        super(BytesSocketClient, self).__init__()
        self._socket = sock
        self._q = queue.Queue()
        self._running = Event()

    def send(self, data: bytes) -> None:
        self._q.put_nowait(data)

    def start(self) -> None:
        logger.info("Starting threaded client")
        self._running.set()
        super(BytesSocketClient, self).start()

    def join(self, timeout: Optional[float] = None) -> None:
        logger.info("Stopping threaded client")
        self._running.clear()
        super(BytesSocketClient, self).join(timeout)

    def __enter__(self) -> BytesSocketClient:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.join()
        super(BytesSocketClient, self).__exit__(exc_type, exc_val, exc_tb)

    def run(self) -> None:
        while self._running.is_set():
            try:
                payload = self._q.get(timeout=0.01)
                self._socket.sendall(payload)
            except queue.Empty:
                continue


class TestCommon(unittest.TestCase):
    @log_test
    def test_stream_packing_frames(self) -> None:
        frames = load_default_trace("test")
        with contextlib.ExitStack() as stack:
            csock, ssock = stack.enter_context(client_server_sockets(timeout=0.250))
            stream = stack.enter_context(
                contextlib.closing(common.frame_stream_unpack(ssock))
            )
            client = stack.enter_context(BytesSocketClient(csock))

            for i in range(frames.step_count):
                logger.debug(f"Sending frame {i}...")
                frame = frames.get_frame(i, "success")
                client.send(common.pack_frame(i + 1, frame))

                # check server side
                logger.debug("Unpacking server side...")
                recv_seq, recv_img = next(stream)
                self.assertEqual(i + 1, recv_seq)
                nptest.assert_array_equal(frame, recv_img)

    @log_test
    def test_packing_responses(self) -> None:
        rng = np.random.default_rng()
        test_image = rng.integers(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
        test_text = "the quick brown fox jumps over the lazy dog"

        with contextlib.ExitStack() as stack:

            csock, ssock = stack.enter_context(client_server_sockets(timeout=0.250))
            stream = stack.enter_context(
                contextlib.closing(common.response_stream_unpack(ssock))
            )
            client = stack.enter_context(BytesSocketClient(csock))

            for t in (True, False):
                logger.debug(f"Sending response with transition: {t}")
                response = (t, test_image, test_text)
                client.send(common.pack_response(*response))
                logger.debug(f"Unpacking server side...")

                recv_t, recv_img, recv_text, recv_size = next(stream)
                self.assertEqual(t, recv_t)
                self.assertEqual(test_text, recv_text)
                nptest.assert_array_equal(test_image, recv_img)


class TestServer(contextlib.AbstractContextManager, Thread):
    def __init__(self, task_name: str, sock: socket.SocketType) -> None:
        super(TestServer, self).__init__()
        self._task_name = task_name
        self._socket = sock
        self._exc_q = queue.Queue()
        self._res_q = queue.Queue()

    def get_exc_queue(self) -> queue.Queue:
        return self._exc_q

    def get_result_queue(self) -> queue.Queue:
        return self._res_q

    def __enter__(self) -> TestServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.join()
        super(TestServer, self).__exit__(exc_type, exc_val, exc_tb)

    def run(self) -> None:
        try:
            server(
                task_name=self._task_name,
                sock=self._socket,
                result_cb=self._res_q.put_nowait,
            )
        except Exception as e:
            logger.exception("Exception in server thread", e)
            self._exc_q.put_nowait(e)


class FrameResultExpecter:
    def __init__(self, testcase: unittest.TestCase):
        self._expected_result = "initial"
        self._test = testcase

    def set_expected(self, expected: str) -> None:
        self._expected_result = expected

    def check_result(self, actual: FrameResult):
        if self._expected_result in ("success", "initial"):
            self._test.assertEqual(FrameResult.SUCCESS, actual)
        elif self._expected_result == "repeat":
            self._test.assertEqual(FrameResult.NO_CHANGE, actual)
        elif self._expected_result == "blank":
            self._test.assertIn(
                actual, (FrameResult.JUNK_FRAME, FrameResult.CV_ERROR, FrameResult)
            )
        else:  # low confidence
            self._test.assertEqual(FrameResult.LOW_CONFIDENCE, actual)


class TestEmulation(unittest.TestCase):
    def test_emulation_loop(self) -> None:
        logger.remove()
        logger.add(sys.stderr, level="INFO", colorize=True)

        # invalid model name
        with self.assertRaises(NotImplementedError):
            StreamSocketEmulation(neuroticism=0.5, trace="test", model="foobar")

        # test all three models
        for model in ("empirical", "theoretical", "naive"):
            logger.info(f"Trying model: {model}")
            emulation = StreamSocketEmulation(
                neuroticism=0.5, trace="test", model=model
            )

            with client_server_sockets(timeout=1.0) as (csock, ssock):
                server_t = TestServer("test", ssock)
                server_t.start()
                expecter = FrameResultExpecter(self)

                def emit_callback(frame: ModelFrame) -> None:
                    expecter.set_expected(frame.frame_tag)
                    try:
                        # callback to check for exceptions in server loop
                        exc = server_t.get_exc_queue().get_nowait()
                        raise exc
                    except queue.Empty:
                        pass

                def result_callback(t: bool, i: Any, s: str) -> None:
                    frame_result = server_t.get_result_queue().get_nowait()
                    expecter.check_result(frame_result)
                    try:
                        # callback to check for exceptions in server loop
                        exc = server_t.get_exc_queue().get_nowait()
                        raise exc
                    except queue.Empty:
                        pass

                emulation.emulate(csock, emit_cb=emit_callback, resp_cb=result_callback)

                # close client first, to make sure no exceptions occur in server
                csock.close()
                try:
                    # callback to check for exceptions in server loop
                    exc = server_t.get_exc_queue().get_nowait()
                    raise exc
                except queue.Empty:
                    pass

                # should be able to join the server thread here, as the client socket
                # closing should have triggered the server loop to end gracefully
                server_t.join(timeout=0.1)
                ssock.close()

                # finally, output the timing metrics dataframe for manual debugging
                logger.info(f"Step metrics:\n{emulation.get_step_metrics()}")
                logger.success("Finished")
