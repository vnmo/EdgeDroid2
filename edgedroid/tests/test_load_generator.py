from __future__ import annotations

import contextlib
import queue
import socket
import sys
import unittest
from functools import wraps
from threading import Event, Thread
from typing import Any, Callable, Iterator, Optional, Tuple

from loguru import logger

from ..data import load_default_trace
from ..load_generator import common
from ..load_generator.client import StreamSocketEmulation


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
    def setUp(self) -> None:
        self.frames = load_default_trace("square00")

    @log_test
    def test_individual_packing_frames(self) -> None:
        for i in range(self.frames.step_count):
            frame_data = self.frames.get_frame(i, "success")
            eframe1 = common.EdgeDroidFrame(i + 1, frame_data)
            eframe2 = common.EdgeDroidFrame.unpack(eframe1.pack())

            self.assertEqual(eframe1, eframe2)

    @log_test
    def test_stream_packing_frames(self) -> None:
        with contextlib.ExitStack() as stack:
            csock, ssock = stack.enter_context(client_server_sockets(timeout=0.250))
            stream = stack.enter_context(
                contextlib.closing(common.frame_stream_unpack(ssock))
            )
            client = stack.enter_context(BytesSocketClient(csock))

            for i in range(self.frames.step_count):
                logger.debug(f"Sending frame {i}...")
                frame_data = self.frames.get_frame(i, "success")
                frame = common.EdgeDroidFrame(i + 1, frame_data)
                client.send(frame.pack())

                # check server side
                logger.debug("Unpacking server side...")
                self.assertEqual(frame, next(stream))

    @log_test
    def test_packing_responses(self) -> None:
        with contextlib.ExitStack() as stack:
            csock, ssock = stack.enter_context(client_server_sockets(timeout=0.250))
            stream = stack.enter_context(
                contextlib.closing(common.response_stream_unpack(ssock))
            )
            client = stack.enter_context(BytesSocketClient(csock))

            for resp in (True, False):
                logger.debug(f"Sending response {resp}...")
                client.send(common.pack_response(resp))
                logger.debug(f"Unpacking server side...")
                self.assertEqual(resp, next(stream))


class DummyServer(contextlib.AbstractContextManager, Thread):
    """
    Simply sends back "true" to each received frame.
    """

    def __init__(self, sock: socket.SocketType) -> None:
        super(DummyServer, self).__init__()
        self._socket = sock
        self._running = Event()

    def start(self) -> None:
        logger.info("Starting dummy server")
        self._running.set()
        super(DummyServer, self).start()

    def join(self, timeout: Optional[float] = None) -> None:
        logger.info("Stopping dummy server")
        self._running.clear()
        super(DummyServer, self).join(timeout)

    def __enter__(self) -> DummyServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.join()
        super(DummyServer, self).__exit__(exc_type, exc_val, exc_tb)

    def run(self) -> None:
        while self._running.is_set():
            try:
                with contextlib.closing(
                    common.frame_stream_unpack(self._socket)
                ) as stream:
                    for _ in stream:
                        self._socket.sendall(common.pack_response(True))
            except socket.timeout:
                pass
            finally:
                self._running.clear()


class TestEmulation(unittest.TestCase):
    def test_emulation_loop(self) -> None:
        logger.remove()
        logger.add(sys.stderr, level="INFO", colorize=True)

        emulation = StreamSocketEmulation(
            neuroticism=0.5, trace="square00", fade_distance=4, model="empirical"
        )

        with contextlib.ExitStack() as stack:
            csock, ssock = stack.enter_context(client_server_sockets(timeout=0.250))
            stack.enter_context(DummyServer(ssock))

            emulation.emulate(csock)
