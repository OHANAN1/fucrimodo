import os
import click
import matplotlib.pyplot as plt
import numpy as np
from fucrimodo.analysis.stage_analysis import (
    get_stage_overview,
    get_modification_overview,
    StageData,
)
from fucrimodo.analysis.utils import get_statistics_overview


def _plot_fitness_stats(stage_data: StageData, row: int):
    fig, ax = plt.subplots()

    results_df = stage_data.fitness_statistics.loc[row, "results"]
    fitness_name = stage_data.fitness_statistics.loc[row, "names"]
    results_df.plot(
        ax=ax,
        x="gen",
        y=["min", "max", "avg"],
        linewidth=2.0,
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel(f"fitness_title")

    ax.set_ylim(0, 1)
    ax.set_xlim(0, stage_data.n_generations)


def main(
    dir_path: str,
    row: int | None = None,
    save_dir: str | None = None,
    show: bool = False,
    verbose: bool = False,
):
    # Load the stage data
    stage_data = StageData(dir_path)

    click.echo("Stage Overview:")
    click.echo(get_stage_overview(stage_data).T)
    click.echo()

    click.echo("-------------------")
    click.echo("Fitness Overview:")
    click.echo(get_statistics_overview(stage_data.fitness_statistics))
    click.echo()

    click.echo("-------------------")
    click.echo("Mutation Overview:")
    click.echo(get_modification_overview(stage_data, "Mutation"))
    click.echo()

    click.echo("-------------------")
    click.echo("Crossover Overview:")
    click.echo(get_modification_overview(stage_data, "Crossover"))
    click.echo()

    # If row is given show the plot or save them to a file
    if row is not None:
        _plot_fitness_stats(stage_data, row)

        if save_dir is not None:
            file_path = f"{save_dir}/stage_statistic_{row}.png"
            plt.savefig(file_path)
            if verbose:
                click.echo(f"Stored file at {file_path}.")
        else:
            plt.show()
