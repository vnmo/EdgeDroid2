import contextlib
import socket
from typing import Callable

from gabriel_lego import FrameResult, LEGOTask

from .common import frame_stream_unpack, pack_response
from .. import data as e_data
from loguru import logger


def _serve(
    task_name: str,
    sock: socket.SocketType,
    result_cb: Callable[[FrameResult], None] = lambda _: None,
) -> None:
    logger.info(f"Starting LEGO task '{task_name}'")
    task = LEGOTask(e_data.load_default_task(task_name))

    with contextlib.closing(frame_stream_unpack(sock)) as frame_stream:
        for frame in frame_stream:
            logger.info(f"Received frame with SEQ {frame.seq}")
            result = task.submit_frame(frame.image_data)
            logger.debug(f"Processing result: {result}")
            result_cb(result)
            if result == FrameResult.SUCCESS:
                logger.success(
                    f"Frame with SEQ {frame.seq} triggers advancement to next step"
                )
                sock.sendall(pack_response(True))
            else:
                sock.sendall(pack_response(False))


def serve_LEGO_task(task: str, port: int, bind_address: str = "0.0.0.0") -> None:
    with socket.create_server(
        (bind_address, port), family=socket.AF_INET, backlog=1, reuse_port=True
    ) as server_sock:
        logger.info(f"Serving LEGO task {task} on {bind_address}:{port}/tcp")
        conn, (peer_addr, peer_port) = server_sock.accept()
        logger.info(f"Opened connection to {peer_addr}:{peer_port}")
        _serve(task, conn)
        logger.warning(f"Closing connection to {peer_addr}:{peer_port}")
        conn.close()
