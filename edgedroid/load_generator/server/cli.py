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
import pathlib
from typing import Optional

import click
import yaml

from .server import serve_LEGO_task
from ..common_cli import enable_logging


@click.command(
    "edgedroid-server",
    context_settings={"auto_envvar_prefix": "EDGEDROID_SERVER"},
)
@click.argument(
    "bind-address",
    type=str,
    envvar="EDGEDROID_SERVER_BIND_ADDR",
)
@click.argument(
    "bind-port",
    type=click.IntRange(0, 65535),
    envvar="EDGEDROID_SERVER_BIND_PORT",
)
@click.argument(
    "task-name",
    type=str,
    envvar="EDGEDROID_SERVER_TASK_NAME",
)
# @click.option(
#     "--one-shot/--multi-run",
#     is_flag=True,
#     default=True,
#     show_default=True,
#     help="Serve a single client and then exit, or stay listening for multiple runs.",
# )
# @click.option(
#     "-o",
#     "--output",
#     type=click.Path(
#         file_okay=True,
#         dir_okay=False,
#         writable=True,
#         resolve_path=True,
#         path_type=pathlib.Path,
#     ),
#     default="./server_records.csv",
#     show_default=True,
#     help="Specifies a path on which to output processed frame metrics in CSV format.",
# )
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    default=None,
    show_default=True,
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
# @click.option(
#     "--log-file",
#     type=click.Path(
#         file_okay=True,
#         dir_okay=False,
#         writable=True,
#         resolve_path=True,
#         path_type=pathlib.Path,
#     ),
#     default=None,
#     show_default=True,
#     help="Save a copy of the logs to a file.",
# )
def edgedroid_server(
    bind_address: str,
    bind_port: int,
    task_name: str,
    # one_shot: bool,
    verbose: bool,
    output_dir: Optional[pathlib.Path],
    # output: pathlib.Path,
    # log_file: Optional[pathlib.Path],
):
    """
    Run an EdgeDroid2 server based on Gabriel-LEGO.

    Binds to BIND-ADDRESS:BIND-PORT and listens for connections from EdgeDroid
    clients. TASK_NAME identifies the task to run.
    """

    # prepare output paths
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        log_file = output_dir / "server.log"
        server_output = output_dir / "server.csv"
    else:
        log_file = None
        server_output = None

    enable_logging(verbose, log_file=log_file)
    serve_LEGO_task(
        task=task_name,
        port=bind_port,
        bind_address=bind_address,
        output_path=server_output,
    )

    if output_dir is not None:
        with (output_dir / "server.metadata.yml").open("w") as fp:
            yaml.safe_dump(
                dict(
                    bind_address=f"{bind_address}:{bind_port}",
                    task=task_name,
                ),
                stream=fp,
                explicit_start=True,
                explicit_end=True,
            )
