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
import loguru
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
    "task",
    type=str,
    envvar="EDGEDROID_SERVER_TRACE",
)
@click.option(
    "--truncate",
    type=int,
    default=-1,
    help="Truncate the specified task trace to a given number of steps. "
    "Note that the client needs to be configured with the same value for the "
    "emulation to work.",
    show_default=False,
)
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
def edgedroid_server(
    bind_address: str,
    bind_port: int,
    task: str,
    truncate: int,
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
    try:
        serve_LEGO_task(
            task_name=task,
            port=bind_port,
            bind_address=bind_address,
            output_path=server_output,
            truncate=truncate,
        )
    except Exception as e:
        loguru.logger.exception(e)
        raise e
    finally:
        if output_dir is not None:
            with (output_dir / "server.metadata.yml").open("w") as fp:
                yaml.safe_dump(
                    dict(
                        bind_address=f"{bind_address}:{bind_port}",
                        task=task,
                    ),
                    stream=fp,
                    explicit_start=True,
                    explicit_end=True,
                )
