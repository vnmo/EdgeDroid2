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
from .server import serve_LEGO_task
from ..common_cli import enable_logging


@click.command("edgedroid-server")
@click.argument("bind-address", type=str)
@click.argument(
    "bind-port",
    type=click.IntRange(0, 65535),
)
@click.argument("task-name", type=str)
@click.option(
    "--one-shot",
    is_flag=True,
    default=True,
    show_default=True,
    help="Serve a single client and then exit.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        writable=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    default="./server_records.csv",
    show_default=True,
    help="Specifies a path on which to output processed frame metrics in CSV format.",
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
def edgedroid_server(
    bind_address: str,
    bind_port: int,
    task_name: str,
    one_shot: bool,
    verbose: bool,
    output: pathlib.Path,
):
    """
    Run an EdgeDroid2 server based on Gabriel-LEGO.

    Binds to BIND-ADDRESS:BIND-PORT and listens for connections from EdgeDroid
    clients. TASK_NAME identifies the task to run.
    """
    enable_logging(verbose)
    serve_LEGO_task(
        task=task_name,
        port=bind_port,
        bind_address=bind_address,
        one_shot=one_shot,
        output_path=output,
    )
