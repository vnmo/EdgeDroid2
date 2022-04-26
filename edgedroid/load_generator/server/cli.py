import click
from .server import serve_LEGO_task
from ..common_cli import set_log_verbosity


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
    "-v",
    "--verbose",
    "log_level",
    type=int,
    count=True,
    default=1,
    help="Enable verbose logging. "
    "Can be specified multiple times for higher levels of debug output.",
    show_default=True,
)
def edgedroid_server(
    bind_address: str, bind_port: int, task_name: str, one_shot: bool, log_level: int
):
    """
    Run an EdgeDroid2 server based on Gabriel-LEGO.

    Binds to BIND-ADDRESS:BIND-PORT and listens for connections from EdgeDroid
    clients. TASK_NAME identifies the task to run.
    """
    set_log_verbosity(log_level)
    serve_LEGO_task(
        task=task_name, port=bind_port, bind_address=bind_address, one_shot=one_shot
    )
