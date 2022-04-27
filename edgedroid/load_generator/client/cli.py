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

import socket
from typing import Literal

import click
from loguru import logger

from .client import StreamSocketEmulation
from ..common_cli import enable_logging


@click.command("edgedroid-client")
@click.argument("host", type=str)
@click.argument("port", type=click.IntRange(0, 65535))
@click.option(
    "-n",
    "--neuroticism",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    show_default=True,
    help="Normalized neuroticism value for the model.",
)
@click.option(
    "-t",
    "--task",
    type=str,
    default="square00",
    show_default=True,
    help="Name of the task to use for emulation.",
)
@click.option(
    "-f",
    "--fade-distance",
    type=int,
    default=8,
    show_default=True,
    help="Distance, in number of steps, after which the model forgets the "
    "most recent transition.",
)
@click.option(
    "-m",
    "--model",
    type=click.Choice(["empirical", "theoretical"], case_sensitive=False),
    default="theoretical",
    show_default=True,
    help="Execution time model to use. "
    "'empirical' samples directly from the underlying data, "
    "'theoretical' first fits distributions to the data and then samples.",
)
@click.option(
    "-v",
    "--verbose",
    type=bool,
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
    show_default=True,
)
def edgedroid_client(
    host: str,
    port: int,
    neuroticism: float,
    task: str,
    fade_distance: int,
    model: Literal["empirical", "theoretical"],
    verbose: bool,
):
    """
    Run an EdgeDroid2 client.

    Connects to HOST:PORT and runs an emulation.
    """

    enable_logging(verbose)
    emulation = StreamSocketEmulation(
        neuroticism=neuroticism, trace=task, fade_distance=fade_distance, model=model
    )

    # "connect" to remote
    # this is of course just for convenience, to skip adding an address to every
    # send() call, as there are no "connections" in udp.
    logger.info(f"Connecting to remote server at {host}:{port}/tcp")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(30.0)  # TODO: magic number, maybe add as an option. This is
        # just the timeout for the initial connection.
        try:
            sock.connect((host, port))
        except socket.timeout:
            logger.error(f"Timed out connecting to backend at {host}:{port}")
            raise click.Abort()
        except ConnectionRefusedError:
            logger.error(f"{host}:{port} refused connection.")
            raise click.Abort()
        except socket.error as e:
            logger.error(
                f"Encountered unspecified socket error when connecting to {host}:{port}"
            )
            logger.exception(e)
            raise click.Abort()

        sock.settimeout(None)  # blocking mode
        # these are tcp sockets, so no timeouts are needed

        emulation.emulate(sock)
