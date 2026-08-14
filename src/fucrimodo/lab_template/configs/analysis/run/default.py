import matplotlib.pyplot as plt
import numpy as np
from fucrimodo.analysis.run_analysis import get_run_overview, RunData
from fucrimodo.analysis.utils import get_statistics_overview
import click


def _plot_global_stats(run_data: RunData, row: int):
    fig, ax = plt.subplots()

    results_df = run_data.global_statistics.loc[row, "results"]
    statistics_name = run_data.global_statistics.loc[row, "names"]
    results_df.plot(
        ax=ax,
        x="gen",
        y=["min", "max", "avg"],
        linewidth=2.0,
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel(statistics_name)

    # Plot a line where the stages change
    stage_ids = results_df["stage_id"].to_numpy()
    change_idx = np.flatnonzero(stage_ids[1:] != stage_ids[:-1])
    ax.vlines(change_idx, -0, 1, "black", zorder=0)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, run_data.n_generations)


def main(
    run_dir: str,
    row: int | None = None,
    save_dir: str | None = None,
    show: bool = True,
    verbose: bool = False,
) -> None:
    run_data = RunData(run_dir)

    run_overview = get_run_overview(run_data)
    global_stats_overview = get_statistics_overview(run_data.global_statistics)

    click.echo("________________________________________________________")
    click.echo("Run Overview:")
    click.echo(run_overview)
    click.echo()
    click.echo("________________________________________________________")
    click.echo("Global Statistics Overview:")
    click.echo(global_stats_overview.T)
    click.echo()
    click.echo()
    click.echo("Note: To properly analyse crystals open ase db cli.")

    # If a row is provided, plot the selected global statistic
    if row is not None:
        _plot_global_stats(run_data, row)

        if save_dir is not None:
            plt.savefig(f"{save_dir}/global_statistic_{row}.png")
            plt.close()
        else:
            plt.show()
