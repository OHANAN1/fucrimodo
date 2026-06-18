import matplotlib.pyplot as plt
from fucrimodo.analysis.run_analysis import (
    get_run_overview,
    get_global_statistics_overview,
    plot_global_statistics,
    RunData
)

def main(
    run_dir: str,
    row: int | None = None,
    save_dir: str | None = None,
    show: bool = True,
    verbose: bool = False,
) -> None:
    run_data = RunData(run_dir)

    print("________________________________________________________")
    print("Run Overview:")
    print(get_run_overview(run_data).T)
    print()
    print("________________________________________________________")
    print("Global Statistics Overview:")
    global_stats_overview = get_global_statistics_overview(run_data)
    print(global_stats_overview.T)
    print()
    print()
    print("Note: To analyse crystals open ase db cli.")

    # If a row is provided, only plot the selected global statistic
    if row is not None:
        plot_global_statistics(run_data, row)
        if show:
            plt.show()
        else:
            if save_dir is not None:
                plt.savefig(f"{save_dir}/global_statistic_{row}.png")
                plt.close()
            else:
                plt.savefig(f"global_statistic_{row}.png")
                plt.close()
