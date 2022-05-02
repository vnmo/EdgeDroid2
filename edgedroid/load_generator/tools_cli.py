from typing import List

import click
from loguru import logger

from .common_cli import enable_logging


def _fetch_traces(trace_names):
    from .. import data as e_data

    logger.info(f"Fetching traces: {trace_names}")
    for trace_name in trace_names:
        e_data.load_default_trace(trace_name=trace_name)


@click.group()
@click.option(
    "-v",
    "--verbose",
    type=bool,
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
    show_default=True,
)
def edgedroid_tools(verbose: bool):
    enable_logging(verbose)


@edgedroid_tools.command()
@click.argument("trace-names", type=str, nargs=-1)
def prefetch_trace(trace_names: List[str]):
    _fetch_traces(trace_names)


@edgedroid_tools.command()
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    default=False,
    type=bool,
    help="Don't prompt for confirmation",
    show_default=True,
)
def prefetch_all_traces(yes: bool):
    from .. import data as e_data

    logger.warning("Prefetching all available traces")
    logger.warning("Note that this can potentially use a lot of disk space!")
    if not yes:
        click.confirm("Continue fetching all traces?", default=False, abort=True)

    _fetch_traces(e_data.load._default_traces.keys())
