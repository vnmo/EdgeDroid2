from typing import List

import click

from common import enable_logging
from edgedroid.trace_fetch import *


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
def trace_fetcher(verbose: bool):
    enable_logging(verbose)


@trace_fetcher.command()
@click.argument("trace-names", type=str, nargs=-1)
def prefetch_trace(trace_names: List[str]):
    fetch_traces(trace_names)


@trace_fetcher.command("prefetch-all")
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
    logger.warning("")
    if not yes:
        click.confirm(
            "Are you sure you wish to pre-fetch all traces? Note that "
            "this can potentially use a lot of disk space!",
            default=False,
            abort=True,
        )

    fetch_all_traces()


if __name__ == "__main__":
    trace_fetcher()
