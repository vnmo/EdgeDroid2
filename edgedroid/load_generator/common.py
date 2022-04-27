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
import struct
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import numpy.typing as npt
from loguru import logger

HEADER_PACK_FMT = "!IIIII"  # [seq][height][width][channels][data length]
HEADER_LEN = struct.calcsize(HEADER_PACK_FMT)
IMG_BYTES_ORDER = "C"

RESP_PACK_FMT = "!?"
RESP_LEN = struct.calcsize(RESP_PACK_FMT)


def bytes_to_numpy_image(
    data: bytes, height: int, width: int, channels: int
) -> npt.NDArray:
    return np.frombuffer(data, dtype=np.uint8).reshape(
        (height, width, channels), order=IMG_BYTES_ORDER
    )


@dataclass
class EdgeDroidFrame:
    seq: int
    image_data: npt.NDArray

    def __eq__(self, other: Any):
        if not isinstance(other, EdgeDroidFrame):
            return False
        else:
            return self.seq == other.seq and (
                np.all(self.image_data == other.image_data)
            )

    def __post_init__(self):
        assert self.image_data.ndim == 3

    def pack(self) -> bytes:
        height, width, channels = self.image_data.shape
        img_data = self.image_data.tobytes(order=IMG_BYTES_ORDER)
        data_len = len(img_data)

        hdr = struct.pack(HEADER_PACK_FMT, self.seq, height, width, channels, data_len)

        return hdr + img_data

    @classmethod
    def unpack(cls, data: bytes) -> EdgeDroidFrame:
        hdr = data[:HEADER_LEN]
        seq, height, width, channels, data_len = struct.unpack(HEADER_PACK_FMT, hdr)
        image_data = bytes_to_numpy_image(
            data[HEADER_LEN : HEADER_LEN + data_len],
            width=width,
            height=height,
            channels=channels,
        )

        return cls(seq, image_data)


def recv_from_socket(sock: socket.SocketType, amount: int) -> bytes:
    received = b""
    while len(received) < amount:
        data = sock.recv(amount - len(received))
        if len(data) == 0:
            # socket is closed
            raise EOFError()
        else:
            received += data
    return received


def frame_stream_unpack(
    sock: socket.SocketType,
) -> Iterator[EdgeDroidFrame]:
    # TODO: does this need to be optimized somehow?
    logger.info("Started frame stream unpacker")
    try:
        while True:
            header = recv_from_socket(sock, HEADER_LEN)
            # got a complete header
            seq, height, width, channels, data_len = struct.unpack(
                HEADER_PACK_FMT, header
            )

            # read data
            image_data = recv_from_socket(sock, data_len)

            # got all the data!
            image = bytes_to_numpy_image(image_data, height, width, channels)
            yield EdgeDroidFrame(seq, image)
    except EOFError:
        logger.warning("Socket was closed")
    finally:
        logger.debug("Closing frame stream unpacker")


def pack_response(resp: bool) -> bytes:
    return struct.pack(RESP_PACK_FMT, resp)


def response_stream_unpack(
    sock: socket.SocketType,
) -> Iterator[bool]:
    try:
        while True:
            resp = recv_from_socket(sock, RESP_LEN)
            yield struct.unpack(RESP_PACK_FMT, resp)[0]  # unpack always returns tuples
    except EOFError:
        logger.warning("Socket was closed")
    finally:
        logger.debug("Closing response stream unpacker")
