import os
import shutil
from collections import deque
from pathlib import Path
from typing import Callable, Collection, Dict, Tuple

import click
import numpy as np
import pandas as pd
import itertools as it
import multiprocess as mp
from tqdm import tqdm

from edgedroid.models import (
    DistExpKernelRollingTTFETModel,
    ExecutionTimeModel,
    ExpKernelRollingTTFETModel,
)
import edgedroid.data as e_data

__all__ = ["models"]

models: Dict[str, Callable[[], ExecutionTimeModel]] = {
    "Low neuro": lambda: ExpKernelRollingTTFETModel(neuroticism=0.0),
    "High neuro": lambda: ExpKernelRollingTTFETModel(neuroticism=1.0),
    "ExGaussian fit,\nlow neuro": lambda: DistExpKernelRollingTTFETModel(
        neuroticism=0.0
    ),
    "ExGaussian fit,\nhigh neuro": lambda: DistExpKernelRollingTTFETModel(
        neuroticism=1.0
    ),
}


def test_model_constant_ttf(
    model_name: str,
    model_const: Callable[[], ExecutionTimeModel],
    ttf_tag: str,
    ttf: float,
    reps: int,
    warmup_steps: int,
    test_steps: int,
    ttf_samples: Collection[float],
) -> pd.DataFrame:
    rng = np.random.default_rng()

    model = model_const()
    rows = deque()
    for rep in range(reps):
        # warmup model
        model.reset()
        for _ in range(warmup_steps):
            model.advance(rng.choice(ttf_samples))

        for i in range(test_steps):
            exec_time = model.advance(ttf).get_execution_time()
            rows.append(
                {
                    "Configuration": model_name,
                    "Time-to-Feedback": ttf_tag,
                    "TTF (Raw)": ttf,
                    "Repetition": rep,
                    "Step": i + 1,
                    "Execution time": exec_time,
                }
            )

    return pd.DataFrame(rows)


def test_model_transitions(
    model_name: str,
    model_const: Callable[[], ExecutionTimeModel],
    origin_ttf_tag: str,
    origin_ttf: float,
    destination_ttf_tag: str,
    destination_ttf: float,
    reps: int,
    warmup_steps: int,
) -> pd.DataFrame:
    model = model_const()
    rows = deque()
    for rep in range(reps):
        # warmup model
        model.reset()
        for _ in range(warmup_steps):
            model.advance(origin_ttf)

        if origin_ttf_tag == destination_ttf_tag:
            transition = "Same level"
        elif origin_ttf < destination_ttf:
            transition = "Lower -> higher"
        else:
            transition = "Higher -> lower"

        # take a single sample in the destination ttf
        exec_time = model.advance(destination_ttf).get_execution_time()
        rows.append(
            {
                "Model": model_name,
                "Origin TTF": origin_ttf_tag,
                "Destination TTF": destination_ttf_tag,
                "Repetition": rep,
                "Execution time": exec_time,
                "Transition": transition,
            }
        )

    return pd.DataFrame(rows)


def test_model_ttf_to_exec_time(
    model_name: str,
    model_const: Callable[[], ExecutionTimeModel],
    ttf_tag: str,
    ttf: float,
    reps: int,
    warmup_steps: int,
    ttf_samples: Collection[float],
) -> pd.DataFrame:
    rng = np.random.default_rng()
    model = model_const()
    neuro = "low" if "low" in model_name.lower() else "high"
    rows = deque()
    for rep in range(reps):
        # warmup model
        model.reset()
        for _ in range(warmup_steps):
            model.advance(rng.choice(ttf_samples))

        # advance the model and take a single sample
        exec_time = model.advance(ttf).get_execution_time()
        rows.append(
            {
                "Model": model_name,
                "Time-to-Feedback": ttf_tag,
                "TTF (Raw)": ttf,
                "Repetition": rep,
                "Execution time": exec_time,
                "Neuroticism": neuro,
            }
        )

    return pd.DataFrame(rows)


@click.command()
@click.option(
    "-r",
    "--reps",
    type=click.IntRange(min=1, min_open=False),
    default=600,
    show_default=True,
)
@click.option(
    "-w",
    "--warmup-steps",
    type=click.IntRange(min=1, min_open=False),
    default=25,
    show_default=True,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=True, file_okay=True, path_type=Path, exists=False),
    default="./verification",
    show_default=True,
)
@click.option(
    "--test-ttfs",
    type=float,
    nargs=3,
    default=(0.0, 2.5, 5.0),
    show_default=True,
)
def generate_data(
    reps: int,
    warmup_steps: int,
    output: Path,
    test_ttfs: Tuple[float, float, float],
):
    if output.exists():
        if click.confirm("Output path already exists. Overwrite?", default=False):
            if output.is_dir():
                shutil.rmtree(output)
            else:
                os.remove(output)
        else:
            raise click.Abort()

    output.mkdir(parents=True, exist_ok=False)

    test_ttfs = {tag: value for tag, value in zip(("low", "medium", "high"), test_ttfs)}
    data, *_ = e_data.load_default_exec_time_data()
    ttf_samples = data["ttf"].to_numpy()

    # generate TTF data
    dfs = deque()
    combs = list(it.product(models.items(), test_ttfs.items()))
    with tqdm(
        total=len(combs), desc="Generating ttf.parquet..."
    ) as bar, mp.Pool() as pool:

        def _callback(df: pd.DataFrame):
            bar.update()
            dfs.append(df)

        def _errback(e: BaseException):
            raise e

        for (name, model_const), (ttf_tag, ttf) in combs:
            pool.apply_async(
                test_model_ttf_to_exec_time,
                args=(name, model_const, ttf_tag, ttf, reps, warmup_steps, ttf_samples),
                callback=_callback,
                error_callback=_errback,
            )

        pool.close()
        pool.join()

    pd.concat(dfs, ignore_index=True).to_parquet(
        output / "ttf.parquet", compression="gzip"
    )
    dfs.clear()

    # generate duration data
    combs = list(it.product(models.items(), test_ttfs.items()))

    with tqdm(
        total=len(combs), desc="Generating duration.parquet..."
    ) as bar, mp.Pool() as pool:

        def _callback(df: pd.DataFrame) -> None:
            dfs.append(df)
            bar.update()

        def _errback(e: BaseException):
            raise e

        for (name, model_const), (ttf_tag, ttf) in combs:
            pool.apply_async(
                test_model_constant_ttf,
                args=(
                    name,
                    model_const,
                    ttf_tag,
                    ttf,
                    reps,
                    warmup_steps,
                    12,
                    ttf_samples,
                ),
                callback=_callback,
                error_callback=_errback,
            )
        pool.close()
        pool.join()

    pd.concat(dfs, ignore_index=True).to_parquet(
        output / "duration.parquet", compression="gzip"
    )
    dfs.clear()

    # generate transition data
    combs = list(it.product(models.items(), test_ttfs.items(), test_ttfs.items()))
    with tqdm(
        total=len(combs), desc="Generating transitions.parquet..."
    ) as bar, mp.Pool() as pool:

        def _callback(df: pd.DataFrame) -> None:
            dfs.append(df)
            bar.update()

        def _errback(e: BaseException):
            raise e

        for (
            (name, model_const),
            (orig_ttf_tag, orig_ttf),
            (dest_ttf_tag, dest_ttf),
        ) in combs:
            pool.apply_async(
                test_model_transitions,
                args=(
                    name,
                    model_const,
                    orig_ttf_tag,
                    orig_ttf,
                    dest_ttf_tag,
                    dest_ttf,
                    reps,
                    warmup_steps,
                ),
                callback=_callback,
                error_callback=_errback,
            )
        pool.close()
        pool.join()

    pd.concat(dfs, ignore_index=True).to_parquet(
        output / "transitions.parquet", compression="gzip"
    )
    click.echo("Done!")


if __name__ == "__main__":
    generate_data()
