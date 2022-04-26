import click
from .server import serve_LEGO_task


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
def edgedroid_server(bind_address: str, bind_port: int, task_name: str, one_shot: bool):
    """
    Run an EdgeDroid2 server based on Gabriel-LEGO.

    Binds to BIND-ADDRESS:BIND-PORT and listens for connections from EdgeDroid
    clients. TASK_NAME identifies the task to run.
    """

    serve_LEGO_task(
        task=task_name, port=bind_port, bind_address=bind_address, one_shot=one_shot
    )
