from fucrimodo.analysis.results_class import RunResults
from fucrimodo.analysis.analyse_run import AnalyseRun
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.analysis.analyse_many_runs import ManyRunsResults, AnalyseManyRuns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import tabulate
import os
import numpy as np

class MultipleManyRunsResults:
    def __init__(
        self, 
        multiple_many_runs_dir: str,
    ) -> None:
        self.multiple_many_runs_dir = multiple_many_runs_dir
        self.multiple_many_runs = self.__load_multiple_many_runs(
            multiple_many_runs_dir
        )

    def __load_multiple_many_runs(
        self, 
        multiple_many_runs_dir: str,
        ignore_dirs: list[str] = ["analysis_results"],
    ) -> list[ManyRunsResults]:
        subdir_paths = []
        for subdir_name in os.listdir(multiple_many_runs_dir):
            subdir_path = os.path.join(multiple_many_runs_dir, subdir_name)
            if os.path.isdir(subdir_path) and subdir_name not in ignore_dirs:
                subdir_paths.append(subdir_path)

        multiple_many_runs = []
        for subdir in subdir_paths:
            many_runs = ManyRunsResults(subdir)
            multiple_many_runs.append(many_runs)

        print(f"Loaded {len(multiple_many_runs)} many run dicts.")

        return multiple_many_runs

class AnalyseMultipleManyRuns:
    def __init__(
        self, 
        multiple_many_runs_results: MultipleManyRunsResults,
        main_stats_key: str = "SimilarityToTargetSOAPFitness_RBFSimilarity",
        main_stat_name: str = "similarity",
        analysis_results_dir_name: str = "analysis_results",
        min_num_stages: int | None = None,
    ) -> None:
        self.multiple_many_runs_results = multiple_many_runs_results

        self.analysis_results_dir = os.path.join(
            self.multiple_many_runs_results.multiple_many_runs_dir,
            analysis_results_dir_name
        )
        os.makedirs(self.analysis_results_dir, exist_ok=True)

        self.main_stats_key = main_stats_key
        self.main_stat_name = main_stat_name

        self.many_runs_analysis_dict: dict[str, AnalyseManyRuns] = {}
        for many_runs in self.multiple_many_runs_results.multiple_many_runs:
            many_runs_analysis = AnalyseManyRuns(
                many_runs_results=many_runs,
                main_stats_key=self.main_stats_key,
                main_stat_name=self.main_stat_name,
                analysis_results_dir_name=analysis_results_dir_name,
                min_num_stages=min_num_stages
            )
            self.many_runs_analysis_dict[many_runs.name] = many_runs_analysis

        self.n_many_runs = len(self.many_runs_analysis_dict)
        print(f"Loaded {self.n_many_runs} many run analysis dicts.")
        # self.analysis_results_dict = self.__get_analysis_results_dict()

    def get_analysis_overview_dict(self) -> dict:
        """
    Returns an overview dict of the analysis.
    Keys:  
    "n_runs": list[int]  
    "n_completed_runs": list[int]  
    "n_runs_same_comp": list[int]  
    "n_runs_same_stoichiometry": list[int]
    "n_runs_in_bounds": list[int]  
    "n_runs_very_similar": list[int]  
    "mean_stat_value": list[float]  
    "mean_stat_only_completed": list[float]  
    "stat_std_only_completed": list[float]
    "mean_n_generations_compl": list[float]
    "mean_n_generations_incompl": list[float]
    "multi_run_name": list[str]
    """
        overview_dict = {}
        for name, many_run_analy in self.many_runs_analysis_dict.items():
            overview_dict_run = many_run_analy.get_analysis_overview_dict()
            for key, value in overview_dict_run.items():
                if key not in overview_dict:
                    overview_dict[key] = []
                overview_dict[key].append(value)

            if "multi_run_name" not in overview_dict:
                overview_dict["multi_run_name"] = []
            overview_dict["multi_run_name"].append(name)

        return overview_dict

    def get_analysis_overview_str(self) -> str:
        overview_dict = self.get_analysis_overview_dict()
        overview_str = f"""
Analysis overview:
\tnumber of multi runs: {len(overview_dict['multi_run_name'])}
\ttotal number of runs: {np.sum(overview_dict['n_runs'])}
\ttotal number of complete runs: {np.sum(overview_dict['n_completed_runs'])}
\ttotal number of runs with similarity > 0.9: {np.sum(overview_dict['n_runs_very_similar'])}
\tmean {self.main_stat_name} best: {overview_dict[f'mean_stat_value']}
\tmean {self.main_stat_name} best completed: {overview_dict[f'mean_stat_only_completed']}
"""

        overview_table = tabulate.tabulate(overview_dict, headers="keys")
        overview_str += "\n" + overview_table
        return overview_str

    def plot_runs_analysis_results_dict_keys(
        self,
        ax: Axes,
        x_key: str,
        y_key: str,
        run_was_completed = True,
        linewidth: int = 2,
        colors: list[str] = ["black", "orange", "green", "blue", "red"],
        size = 300,
        marker: str = "o",
        edgecolor = "gray",
        alpha: float | int = 1,
        zorder: int = 1,
    ) -> None:
        """
        Plot the values of the keys of the analysis results dict of all runs.
        """
        for i, many_run_analy in enumerate(self.many_runs_analysis_dict.values()):
            many_run_analy.plot_analysis_results_dict_keys(
                ax=ax,
                x_key=x_key,
                y_key=y_key,
                marker=marker,
                run_was_completed=run_was_completed,
                color=colors[i],
                linewidth=linewidth,
                alpha=alpha,
                size=size,
                edgecolor=edgecolor,
                zorder=zorder,
            )

    def plot_stats_key_hist(
        self,
        ax: Axes,
        key: str | None = None,
        run_was_completed: bool = True,
        colors: list[str] = ["black", "orange", "green", "blue", "red"],
        alpha: float = 0.5,
        bins: list[float] | np.ndarray = np.arange(0, 1.1, 0.1),
        bin_width_total: float = 0.09,
        add_text: bool = True,
        add_legend: bool = True,
        legend_params: dict = dict(
            bbox_to_anchor=(0.5, 1.05), loc="lower center", fontsize=20
        )
    ):
        """
        Plots a histogram of the stats key. If key is None, it will plot 
        the key: "best_{self.main_stat_name}".
        """
        if len(colors) < self.n_many_runs:
            raise ValueError(
                "Not enough colors to plot all runs."
            )

        bin_width = bin_width_total / self.n_many_runs

        for i, many_run_analy in enumerate(
            self.many_runs_analysis_dict.values()
        ):
            many_run_analy.plot_stats_key_hist(
                ax=ax,
                key=key,
                run_was_completed=run_was_completed,
                color=colors[i],
                alpha=alpha,
                bins=bins,
                bin_width=bin_width,
                bin_edge_offset= - bin_width + i * bin_width,
                add_text=add_text
            )

        if add_legend:
            legend_params["ncol"] = self.n_many_runs
            ax.legend(**legend_params)



# ╔══════════════════════════════════════════════════════════╗
# ║                    Analysis Functions                    ║
# ╚══════════════════════════════════════════════════════════╝

def create_analysis_plot(
    analysis_multiple_many_runs: AnalyseMultipleManyRuns,
    y_key: str,
    x_key: str,
    ax: list[Axes] | None = None,
    y_label: str | None = None,
    x_label: str | None = None,
    size: int = 700,
    linewidth: int = 1,
    marker: str = "o",
    alpha: float | int = .7,
    save_fig: bool = False,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    show_legend: bool = False,
    legend_params: dict = dict(
        bbox_to_anchor=(0.5, 1.05), loc="lower center", fontsize=20
    ),
    figsize: tuple[int|float, int|float] = (8, 10),
    x_ticks: list[float | str] | None = None,
    x_tick_labels: list[str] | None = None,
    y_ticks: list[float | str] | None = None,
    y_tick_labels: list[str] | None = None,
    gridspec_kw: dict | None = {"hspace": 0.1, "wspace": 0.1, "left": 0.3, "right": 0.95, "top": 0.95, "bottom": 0.15},
) -> None:
    if y_label is None:
        y_label = y_key
    if x_label is None:
        x_label = x_key

    if ax is None:
        if gridspec_kw is not None:
            fig, axes = plt.subplots(
                nrows=2, ncols=1, figsize=figsize, gridspec_kw=gridspec_kw, sharex=True
            )
        else:
            fig, axes = plt.subplots(
                nrows=2, ncols=1, figsize=figsize, tight_layout=True, sharex=True
            )
        assert type(axes) == np.ndarray, "Somehow axes where initialized falsely"

        ax_compl: Axes = axes[0]
        ax_incompl: Axes = axes[1]
    else:
        ax_compl: Axes = ax[0]
        ax_incompl: Axes = ax[1]

    ax_compl.set_ylabel(y_label)
    ax_compl.grid(True)
    analysis_multiple_many_runs.plot_runs_analysis_results_dict_keys(
        ax=ax_compl,
        y_key=y_key,
        x_key=x_key,
        linewidth=linewidth,
        marker=marker,
        alpha=alpha,
        size=size,
        run_was_completed=True,
    )
    
    ax_incompl.set_xlabel(x_label)
    ax_incompl.set_ylabel(y_label)
    ax_incompl.grid(True)
    analysis_multiple_many_runs.plot_runs_analysis_results_dict_keys(
        ax=ax_incompl,
        y_key=y_key,
        x_key=x_key,
        linewidth=linewidth,
        marker=marker,
        alpha=alpha,
        size=size,
        run_was_completed=False,
    )

    if x_lim is not None:
        ax_compl.set_xlim(x_lim)
        ax_incompl.set_xlim(x_lim)
    if y_lim is not None:
        ax_compl.set_ylim(y_lim)
        ax_incompl.set_ylim(y_lim)

    if x_ticks is not None:
        ax_incompl.set_xticks(x_ticks)
    if x_tick_labels is not None:
        ax_incompl.set_xticklabels(x_tick_labels)


    if y_ticks is not None:
        ax_compl.set_yticks(y_ticks)
        ax_incompl.set_yticks(y_ticks)
    if y_tick_labels is not None:
        ax_compl.set_yticklabels(y_tick_labels)
        ax_incompl.set_yticklabels(y_tick_labels)

    if show_legend:
        legend_params["ncol"] = analysis_multiple_many_runs.n_many_runs
        ax_compl.legend(**legend_params)

    if save_fig:
        save_path = os.path.join(
            analysis_multiple_many_runs.analysis_results_dir, 
            f"{y_key}_vs_{x_key}.png"
        )
        plt.savefig(save_path)
        plt.close()

        print(f"Saved plot to {save_path}")


def create_stats_key_hist(
    analysis_multiple_many_runs: AnalyseMultipleManyRuns,
    key: str | None = None,
    colors_completed: list[str] = ["black", "orange", "green", "blue", "red"],
    colors_incomplete: list[str] = ["black", "orange", "green", "blue", "red"],
    alpha_completed: float = 0.5,
    alpha_incomplete: float = 0.5,
    bins: list[float] | np.ndarray = np.arange(0, 1.1, 0.1),
    bin_width_total: float = 0.09,
    add_text: bool = True,
    save_fig: bool = False,
    x_label: str | None = "Similarity",
    y_label: str | None = "N$_{\\text{runs}}$",
    legend_params: dict = dict(
        bbox_to_anchor=(0.4, 1.05), loc="lower center", fontsize=30, labelspacing=0.1, columnspacing=0.3, handletextpad=0.1
    ),
    gridspec_kw: dict | None = {"hspace": 0.1, "wspace": 0.2, "left": 0.1, "right": 0.95, "top": 0.80, "bottom": 0.2},
) -> None:
    """
    Create a histogram of the highest similarity values of all runs.
    If key is None, it will plot the key: "best_{self.main_stat_name}".
    """

    if gridspec_kw is not None:
        fig, ax = plt.subplots(
            nrows=1, ncols=2, figsize=(18, 7), sharex=True, gridspec_kw=gridspec_kw
        )
    else:
        fig, ax = plt.subplots(
            nrows=1, ncols=2, figsize=(18, 7), sharex=True, tight_layout=True
        )

    assert type(ax) == np.ndarray, "Somehow axes where initialized falsely"
    ax_comp: Axes = ax[0]
    analysis_multiple_many_runs.plot_stats_key_hist(
        ax=ax_comp,
        key=key,
        run_was_completed=True,
        colors=colors_completed,
        alpha=alpha_completed,
        bins=bins,
        bin_width_total=bin_width_total,
        add_text=add_text,
        add_legend=False
    )
    ax_comp.grid(axis="y")
    ax_comp.set_xlim(- bin_width_total/2, 0.9 + bin_width_total/2)
    ax_comp.set_xticks(np.arange(0, 1.0, 0.1))
    ax_comp.set_xticklabels(["0.1", "", "0.3", "", "0.5", "", "0.7", "", "0.9", ""])
    if x_label is not None:
        ax_comp.set_xlabel(x_label)

    if y_label is not None:
        ax_comp.set_ylabel(y_label)

    ax_incomp: Axes = ax[1]
    analysis_multiple_many_runs.plot_stats_key_hist(
        ax=ax_incomp,
        key=key,
        run_was_completed=False,
        colors=colors_incomplete,
        alpha=alpha_incomplete,
        bins=bins,
        bin_width_total=bin_width_total,
        add_text=add_text,
        add_legend=True,
        legend_params=legend_params
    )
    ax_incomp.grid(axis="y")
    ax_incomp.set_xlim(ax_comp.get_xlim())
    ax_incomp.set_xticks(ax_comp.get_xticks())
    ax_incomp.set_xticklabels(ax_comp.get_xticklabels())
    if x_label is not None:
        ax_incomp.set_xlabel(x_label)

    # Run specific, I will fix that later
    ax_comp.set_yticks([0, 3, 6, 9, 12])
    ax_incomp.set_yticks([0, 10, 20, 30, 40])

    if save_fig:
        save_path = os.path.join(
            analysis_multiple_many_runs.analysis_results_dir, 
            f"hist_{key}.png"
        )
        plt.savefig(save_path)
        plt.close()

        print(f"Saved plot to {save_path}")

def create_latex_overview_table(
    analysis_multiple_many_runs: AnalyseMultipleManyRuns,
    keys_to_include: list[str] = [
        "multi_run_name", 
        "n_completed_runs", 
        "n_runs_very_similar", 
        "mean_stat_only_completed", 
        "stat_std_only_completed", 
        "n_runs_same_comp", 
        "n_runs_same_stoich", 
    ],
    headers: list[str] | None = [
        r"Database", 
        r"N$_{\text{fin}}$",
        r"N$_{\text{S} > 0.9 ,\text{fin}}$",
        r"$\overline{\text{S}}_{\text{fin}}$",
        r"$\sigma($S$_{\text{fin}})$",
        r"N$_{\text{comp., fin.}}$",
        r"N$_{\text{stoich., fin.}}$",
    ],
) -> None:
    """
    Create a latex table with the overview of the analysis.
    If headers is None, the keys_to_include will be used as headers.

    All keys in keys_to_include must be in the overview dict.
    """
    if headers is None:
        headers = keys_to_include

    overview_dict = analysis_multiple_many_runs.get_analysis_overview_dict()
    for key in keys_to_include:
        if key not in overview_dict:
            raise ValueError(
                f"Key {key} not found in overview dict. "
                f"Available keys: {overview_dict.keys()}"
            )

    overview_table = { key: overview_dict[key] for key in keys_to_include }
    overview_table_latex = tabulate.tabulate(
        overview_table, headers=headers, tablefmt="latex_raw"
    )

    save_path = os.path.join(
        analysis_multiple_many_runs.analysis_results_dir, 
        "overview_table.tex"
    )
    with open(save_path, "w") as f:
        f.write(overview_table_latex)


# ╔══════════════════════════════════════════════════════════╗
# ║                      Example Usage                       ║
# ╚══════════════════════════════════════════════════════════╝

def main(folder_path: str) -> None:
    multiple_many_runs = MultipleManyRunsResults(folder_path)
    analysis = AnalyseMultipleManyRuns(
        multiple_many_runs,
        min_num_stages=4,
    )

    overview_dict = analysis.get_analysis_overview_dict()
    print(overview_dict.keys())

    plt.rcParams.update({'font.size': 50})

    create_analysis_plot(
        analysis,
        x_key="density_target",
        x_label="$\\rho_{\\text{target}}$",
        y_key="density_best",
        y_label="$\\rho_{\\text{best}}$",
        save_fig=True,
        x_lim=(0., 0.12),
        y_lim=(0., 0.12),
        x_ticks=[0., 0.05, 0.10],
        x_tick_labels=["0", "0.05", "0.10"],
        y_ticks=[0., 0.05, 0.10],
        y_tick_labels=["", "0.05", "0.10"],
    )

    create_analysis_plot(
        analysis,
        x_key="density_target",
        x_label="$\\rho_{\\text{target}}$",
        y_key=f"best_{analysis.main_stat_name}",
        y_label="S$_{\\text{max}}$",
        save_fig=True,
        x_lim=(0., 0.12),
        y_lim=(-0.1, 1.1),
        x_ticks=[0., 0.05, 0.10],
        x_tick_labels=["0", "0.05", "0.10"],
        y_ticks=[0., 0.5, 1.0],
        y_tick_labels=["0", "0.5", "1.0"],
    )

    create_analysis_plot(
        analysis,
        x_key="n_generations",
        x_label="N$_{\\text{g}}$",
        y_key=f"best_{analysis.main_stat_name}",
        y_label="S$_{\\text{max}}$",
        save_fig=True,
        x_lim=(0, 550),
        y_lim=(-0.1, 1.1),
        y_ticks=[0., 0.5, 1.0],
        y_tick_labels=["0", "0.5", "1.0"],
        x_ticks=[0, 250, 500],
    )

    # create_analysis_plot(
    #     analysis,
    #     x_key="n_generations",
    #     x_label="N$_{\\text{g}}$",
    #     y_key=f"best_{analysis.main_stat_name}",
    #     y_label="S$_{\\text{max}}$",
    #     save_fig=False,
    #     show_legend=True,
    #     legend_params=dict(
    #         bbox_to_anchor=(0.30, 1.05), loc="lower center", fontsize=30,
    #         columnspacing=0.2, handletextpad=0.1
    #     ),
    #     x_lim=(0, 550),
    #     y_lim=(-0.1, 1.1),
    #     figsize=(9, 11.0),
    #     y_ticks=[0., 0.5, 1.0],
    #     y_tick_labels=["0", "0.5", "1.0"],
    #     x_ticks=[0, 250, 500],
    #     gridspec_kw={"hspace": 0.1, "wspace": 0.1, "left": 0.3, "right": 0.95, "top": 0.85, "bottom": 0.15},
    # )

    # create_analysis_plot(
    #     analysis,
    #     x_key="n_generations",
    #     x_label="N$_{\\text{g}}$",
    #     y_key=f"density_target",
    #     y_label="$\\rho_{\\text{target}}$",
    #     save_fig=True,
    #     x_lim=(0, 550),
    #     y_lim=(0.0, 0.12),
    #     y_ticks=[0., 0.05, 0.10],
    #     y_tick_labels=["", "0.05", "0.10"],
    #     x_ticks=[0, 250, 500],
    # )


    target_densities_completed = []
    for many_run_name, many_run_analy in analysis.many_runs_analysis_dict.items():
        analysis_results_dict = many_run_analy.get_analysis_results_dict()
        for target_density, completed in zip(
            analysis_results_dict["density_target"],
            analysis_results_dict["was_completed"]
        ):
            if completed:
                target_densities_completed.append(target_density)

    print(f"Max density target completed runs: {np.max(target_densities_completed)}")

    analysis_overview_str = analysis.get_analysis_overview_str()
    print(analysis_overview_str)

    create_latex_overview_table(analysis)

    plt.rcParams.update({'font.size': 35})
    create_stats_key_hist(
        analysis,
        key=None,
        alpha_completed=0.5,
        alpha_incomplete=0.5,
        bins=np.arange(0, 1.1, 0.1),
        bin_width_total=0.09,
        add_text=True,
        save_fig=True,
        x_label="Maximum similarity S$_{\\text{max}}$",
    )
    # create_analysis_plot(
    #     analysis,
    #     x_key="volume_target",
    #     y_key=f"best_{analysis.main_stat_name}",
    #     save_fig=True,
    # )
    # create_analysis_plot(
    #     analysis,
    #     x_key="n_atoms_target",
    #     y_key=f"best_{analysis.main_stat_name}",
    #     save_fig=True,
    # )

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} path/to/run/dir")
        sys.exit(1)

    folder_path = sys.argv[1]

    main(folder_path)


