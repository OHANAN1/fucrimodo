import warnings
from matplotlib.axes import Axes
import os
import ase
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from tabulate import tabulate
from icecream import ic
import matplotlib.pyplot as plt
from src.utils.analyse_results.results_class import RunResults
from src.utils.analyse_results.analyse_run import AnalyseRun
from src.fucrimodo.utils.cellbounds_custom import CustomCellBounds

from datetime import datetime

ic.enable()
warnings.filterwarnings("default")

# Data Class

class ManyRunsResults():
    def __init__(
        self, 
        runs_dir: str,
        ignore_dirs: list[str] = ["analysis_results"],
        name: str | None = None,
    ) -> None:
        self.runs_dir = runs_dir
        self.runs = self.__load_runs(
            runs_dir, ignore_dirs=ignore_dirs
        )
        self.n_runs = len(self.runs)
        ic(f"Found {self.n_runs} runs.")

        if name is None:
            self.name = os.path.basename(runs_dir)

    def __load_runs(
        self, 
        runs_dir: str,
        ignore_dirs: list[str] = [],
    ) -> list[RunResults]:
        subdir_paths = []
        for subdir_name in os.listdir(runs_dir):
            subdir_path = os.path.join(runs_dir, subdir_name)
            if os.path.isdir(subdir_path) and subdir_name not in ignore_dirs:
                subdir_paths.append(subdir_path)

        runs = []
        for subdir in subdir_paths:
            run = RunResults(subdir)
            runs.append(run)

        return runs

# Main classes
class AnalyseManyRuns():
    """
    Class to analyse the results of many runs.
    Set the main stats key and the main stat name to analyse it.
    Needs to be present in all runs.

    Use max number of stages if you want to compare the runs and some 
    have less stages than others bc they were stopped early.
    If set to none, it will use the maximum number of stages of all runs.

    Set sort_runs_by_names to sort the runs by the names in the list.
    """
    def __init__(
        self, 
        many_runs_results: ManyRunsResults,
        main_stats_key: str = "SimilarityToTargetSOAPFitness_RBFSimilarity",
        main_stat_name: str = "similarity",
        analysis_results_dir_name: str = "analysis_results",
        cell_bounds: CustomCellBounds | None = None,
        min_num_stages: int | None = None,
        sort_runs_by_names: list[str] | None = None,
    ) -> None:
        self.many_runs_results = many_runs_results

        if sort_runs_by_names is not None:
            sorted_runs = []
            for run_name in sort_runs_by_names:
                for run in self.many_runs_results.runs:
                    if run_name == run.run_name:
                        sorted_runs.append(run)
                        break
            self.many_runs_results.runs = sorted_runs

            print("Sorted runs:")
            for run in self.many_runs_results.runs:
                print(run.run_name)


        self.analysis_results_dir = os.path.join(
            many_runs_results.runs_dir, analysis_results_dir_name
        )
        if not os.path.exists(self.analysis_results_dir):
            os.mkdir(self.analysis_results_dir)
        ic(self.analysis_results_dir)

        self.main_stats_key = main_stats_key
        self.main_stat_name = main_stat_name
        ic(main_stats_key)
        ic(main_stat_name)

        self.cell_bounds = cell_bounds

        min_num_stages = 0
        self.runs_analysis_dict: dict[str, AnalyseRun] = {}
        for run in self.many_runs_results.runs:
            print(f"Analyse run {run.run_name}")
            run_name = run.run_name
            run_analysis = AnalyseRun(
                run_results=run,
                main_stats_key=self.main_stats_key,
                main_stat_name=self.main_stat_name,
                cell_bounds=self.cell_bounds,
                min_num_stages=min_num_stages,
            )
            self.runs_analysis_dict[run_name] = run_analysis

            if min_num_stages is None:
                num_stages = run_analysis.get_number_of_stages()
                if num_stages > min_num_stages:
                    min_num_stages = num_stages

        self.min_num_stages = min_num_stages
        self.name = self.many_runs_results.name

        self.analysis_results_dict = self.get_analysis_results_dict()

    def get_min_number_of_stages_all_runs(self) -> int:
        """
        Returns the maximum number of stages of all runs.
        """
        max_n_stages = 0
        for run_analysis in self.runs_analysis_dict.values():
            n_stages = run_analysis.get_number_of_stages()
            if n_stages > max_n_stages:
                max_n_stages = n_stages

        return max_n_stages

    def get_best_crystals_dict(
        self
    ) -> dict[str, list[ase.Atoms | float | int | str | None]]:
        """
        Returns a dictionary with the best crystal tuple of each run.
        dict keys: 
        "run_name": str
        "best_crystal": ase.Atoms
        "best_crystal_value": float
        "stage_id": int

        or None for everything but run name if no stages are present.
        """
        best_crystals_dict = {
            "run_name": [],
            "best_crystal": [],
            "best_crystal_value": [],
            "stage_id": [],
        }

        for run_name, run_analysis in self.runs_analysis_dict.items():
            best_crystals_dict["run_name"].append(run_name)
            best_crystal_tupel = run_analysis.get_best_crystal_tuple(
                statistics_key=self.main_stats_key
            )
            if best_crystal_tupel is not None:
                crystal, stat_value, stage_id = best_crystal_tupel
            else:
                crystal, stat_value, stage_id = None, None, None

            best_crystals_dict["best_crystal"].append(crystal)
            best_crystals_dict["best_crystal_value"].append(stat_value)
            best_crystals_dict["stage_id"].append(stage_id)

        return best_crystals_dict

    # def __add_analysis_dict_entry(
    #     self,
    #     run_name: str,
    #     was_completed: bool,
    #     target_crystal: ase.Atoms,
    #     best_crystal: ase.Atoms | None,
    #     best_crystal_value: float | None,
    #     stage_id: int | None | str,
    #     same_comp: bool | None,
    #     same_stoichi: bool | None,
    #     ratio: float | str | None,
    #     n_gen_each_stage: dict[int, int],
    #     best_value_each_stage: dict[int, float],
    #     analysis_results_dict: dict = {},
    #     round_values: int = 3,
    #     n_generations: int | None = None,
    # ) -> dict:
    #     if analysis_results_dict == {}:
    #         analysis_results_dict = {
    #             "run_name": [],
    #             "was_completed": [],
    #             "target_crystal": [],
    #             "best_crystal": [],
    #             "found_in_stage": [],
    #             f"best_{self.main_stat_name}": [],
    #             "same_composition": [],
    #             "same_stoichiometry": [],
    #             "ratio": [],
    #             "target_in_bounds": [],
    #             "n_generations": [],
    #             "density_target": [],
    #             "density_best": [],
    #         }
    #         for stage_index in range(self.max_n_stages):
    #             stage_id = stage_index + 1
    #             analysis_results_dict[f"n_gen_stage_{stage_id}"] = []
    #             analysis_results_dict[f"best_value_stage_{stage_id}"] = []
    #
    #     if isinstance(best_crystal_value, float):
    #         best_crystal_value = round(best_crystal_value, round_values)
    #
    #     analysis_results_dict["run_name"].append(run_name)
    #     analysis_results_dict["was_completed"].append(was_completed)
    #
    #     analysis_results_dict["target_crystal"].append(target_crystal)
    #     analysis_results_dict["density_target"].append(
    #         len(target_crystal) / target_crystal.get_volume()
    #     )
    #
    #     analysis_results_dict["best_crystal"].append(best_crystal)
    #     if best_crystal is not None:
    #         analysis_results_dict["density_best"].append(
    #             len(best_crystal) / best_crystal.get_volume()
    #         )
    #
    #     analysis_results_dict["found_in_stage"].append(stage_id)
    #     analysis_results_dict[f"best_{self.main_stat_name}"].append(
    #         best_crystal_value
    #     )
    #     analysis_results_dict["same_composition"].append(same_comp)
    #     analysis_results_dict["same_stoichiometry"].append(same_stoichi)
    #     analysis_results_dict["ratio"].append(ratio)
    #
    #     analysis_results_dict["n_generations"].append(n_generations)
    #
    #     if self.cell_bounds is not None:
    #         bounds = self.cell_bounds.is_within_bounds(target_crystal.cell)
    #         analysis_results_dict["target_in_bounds"].append(bounds)
    #     else:
    #         analysis_results_dict["target_in_bounds"].append("-")
    #
    #     for stage_id in range(1, self.max_n_stages + 1):
    #         if stage_id in n_gen_each_stage.keys():
    #             n_gens = n_gen_each_stage[stage_id]
    #         else:
    #             n_gens = None
    #
    #         analysis_results_dict[f"n_gen_stage_{stage_id}"].append(n_gens)
    #
    #         if stage_id in best_value_each_stage.keys():
    #             best_value = best_value_each_stage[stage_id]
    #             if isinstance(best_value, float):
    #                 best_value = round(best_value, round_values)
    #         else:
    #             best_value = None
    #
    #         analysis_results_dict[f"best_value_stage_{stage_id}"].append(
    #             best_value
    #         )
    #
    #     return analysis_results_dict

    def get_analysis_results_dict(
        self,
        round_values: int = 3,
    ) -> dict:
        """
        Returns a dictionary of the best crystals.
        Keys:
        "run_name": list[str]
        "was_completed": list[bool]
        "target_crystal": list[aes.Atoms]
        "best_crystal": list[ase.Atoms | None]
        "best_crystal_{main_stat_name}": list[float]
        "found_in_stage": list[int]
        "same_composition": list[bool]
        "same_stoichiometry": list[bool]
        "ratio": list[float]
        "n_generations": list[int]
        "target_in_bounds": list[bool]
        "density_target": list[float]
        "density_best": list[float]
        "n_atoms_target": int
        "n_atoms_best": int
        "volume_target": float
        "volume_best": float
        "n_gen_stage_{stage_id}": list[dict[int, int]]
        "best_value_stage_{stage_id}": list[dict[int, float]]
        """
        if hasattr(self, "analysis_results_dict"):
            return self.analysis_results_dict

        analysis_results_dict = {}
        for run_name, single_run_analysis in self.runs_analysis_dict.items():
            run_analysis = single_run_analysis.get_analysis_results_dict(
                round_values=round_values
            )
            for key, value in run_analysis.items():
                if key not in analysis_results_dict.keys():
                    analysis_results_dict[key] = []
                analysis_results_dict[key].append(value)

        #ic(analysis_results_dict.keys())

            # best_crystal_tuple = run_analysis.get_best_crystal_tuple(
            #     statistics_key=self.main_stats_key
            # )
            # same_stoichi, ratio = run_analysis.target_and_best_have_same_stoichiometry()
            # same_comp = run_analysis.target_and_best_have_same_composition()
            #
            # if best_crystal_tuple is not None:
            #     best_crystal, best_crystal_value, stage_id = best_crystal_tuple
            # else:
            #     best_crystal, best_crystal_value, stage_id = None, None, None
            #
            # analysis_results_dict = self.__add_analysis_dict_entry(
            #     run_name=run_name,
            #     was_completed=run_analysis.was_completed(),
            #     target_crystal=run_analysis.target_crystal,
            #     best_crystal=best_crystal,
            #     best_crystal_value=best_crystal_value,
            #     stage_id=stage_id,
            #     same_comp=same_comp,
            #     same_stoichi=same_stoichi,
            #     ratio=ratio,
            #     round_values=round_values,
            #     analysis_results_dict=analysis_results_dict,
            #     n_generations=run_analysis.get_n_generations(),
            #     n_gen_each_stage=run_analysis.get_n_generations_each_stage(),
            #     best_value_each_stage=run_analysis.get_best_value_each_stage(),
            # )
        return analysis_results_dict

    def get_analysis_results_dict_keys(self) -> list[str]:
        """
        Returns the keys of the analysis results dict.
        Keys can be used in plot and anywhere.
        """
        return list(self.get_analysis_results_dict().keys())

    def get_analysis_results_table(
        self,
        round_values: int = 3,
        output_type: str = "plain",
    ) -> str | dict:
        """
        Returns a tabulated string of the best crystals.
        output_types: 
        "plain": returns a plain text table
        "latex": returns a latex table
        "dict": returns the dictionary
        """
        analysis_results_table = self.get_analysis_results_dict(
            round_values=round_values
        )

        best_crystals_formula = []
        for best_crystal in analysis_results_table["best_crystal"]:
            if best_crystal is not None:
                best_crystals_formula.append(best_crystal.get_chemical_formula())
            else:
                best_crystals_formula.append(None)
        analysis_results_table["best_crystal"] = best_crystals_formula

        target_crystals_formula = []
        for target_crystal in analysis_results_table["target_crystal"]:
            target_crystals_formula.append(target_crystal.get_chemical_formula())
        analysis_results_table["target_crystal"] = target_crystals_formula

        if output_type == "latex":
            table = tabulate(
                analysis_results_table, headers="keys", tablefmt="latex"
            )
            return table

        elif output_type == "dict":
            return analysis_results_table

        elif output_type == "plain":
            table = tabulate(
                analysis_results_table, headers="keys"
            )
            return table
        
        else:
            raise ValueError(
                f"Output type {output_type} not recognized."
                " Choose from 'plain', 'latex' or 'dict'."
            )

    def get_analysis_overview_dict(
        self,
        round_values: int = 3,
    ) -> dict[str, int | float | str | None | bool]:
        """
        Returns an overview dict of the analysis.
        Keys:
        "n_runs": int
        "n_completed_runs": int
        "n_runs_same_comp": int
        "n_runs_same_comp_completed": int
        "n_runs_same_stoich": int
        "n_runs_same_stoich_completed": int
        "n_runs_in_bounds": int
        "n_runs_very_similar": int
        "mean_stat_value": float
        "mean_stat_only_completed": float
        "stat_std_only_completed": float
        "mean_n_generations_compl": float
        "mean_n_generations_incompl": float
        """
        analysis_results_dict = self.get_analysis_results_dict(
            round_values=round_values,
        )

        overview_dict = {}
        overview_dict["n_runs"] = len(analysis_results_dict["run_name"])
        overview_dict["n_completed_runs"] = np.sum(
            np.array(analysis_results_dict["was_completed"]) == True
        )
        overview_dict["n_runs_same_comp"] = np.sum(
            np.array(np.array(analysis_results_dict["same_composition"]) == True)
        )
        overview_dict["n_runs_same_comp_completed"] = np.sum(
            np.array(
                np.array(analysis_results_dict["same_composition"]) == True
            )[np.array(analysis_results_dict["was_completed"]) == True]
        )
        overview_dict["n_runs_same_stoich"] = np.sum(
            np.array(np.array(analysis_results_dict["same_stoichiometry"]) == True)
        )
        overview_dict["n_runs_same_stoich_completed"] = np.sum(
            np.array(
                np.array(analysis_results_dict["same_stoichiometry"]) == True
            )[np.array(analysis_results_dict["was_completed"]) == True]
        )
        overview_dict["n_runs_in_bounds"] = np.sum(
            [
                1 for bound in analysis_results_dict["target_in_bounds"] 
                if bound
            ]
        )
        n_runs_very_similar = 0
        for value in analysis_results_dict[f"best_{self.main_stat_name}"]:
            if isinstance(value, float):
                if value >= 0.9:
                    n_runs_very_similar += 1
        overview_dict["n_runs_very_similar"] = n_runs_very_similar

        overview_dict["mean_stat_value"] = np.mean(
            np.array(analysis_results_dict[f"best_{self.main_stat_name}"])
        )
        overview_dict["mean_stat_only_completed"] = np.mean(
            np.array(
                analysis_results_dict[f"best_{self.main_stat_name}"]
            )[np.array(analysis_results_dict["was_completed"]) == True]
        )
        overview_dict["stat_std_only_completed"] = np.std(
            np.array(
                analysis_results_dict[f"best_{self.main_stat_name}"]
            )[np.array(analysis_results_dict["was_completed"]) == True]
        )
        overview_dict["mean_n_generations_compl"] = np.mean(
            np.array(
                analysis_results_dict["n_generations"]
            )[np.array(analysis_results_dict["was_completed"]) == True]
        )
        overview_dict["mean_n_generations_incompl"] = np.mean(
            np.array(
                analysis_results_dict["n_generations"]
            )[np.array(analysis_results_dict["was_completed"]) == False]
        )

        return overview_dict

    def get_analysis_overview_str(
        self,
        round_values: int = 3,
    ) -> str:
        """
        Returns an overview of the analysis.
        """
        overview_dict = self.get_analysis_overview_dict(
            round_values=round_values
        )

        return f"""
Number of runs: {overview_dict["n_runs"]}
Number of completed runs: {overview_dict["n_completed_runs"]}
Number of runs with same composition: {overview_dict["n_runs_same_comp"]}
Number of runs with target in bounds: {overview_dict["n_runs_in_bounds"]}
Number of runs with similarity > 0.9: {overview_dict["n_runs_very_similar"]}
Mean {self.main_stat_name}: {overview_dict["mean_stat_value"]}
Mean {self.main_stat_name} of completed runs: {overview_dict["mean_stat_only_completed"]}
        """

    def plot_combined_statistics_development(
        self,
        ax: Axes,
        display_stage_id: bool = True,
        stage_id_y_pos: float = 1.,
        stage_id_x_offset: float = 2.5,
        value_types = ["max", "min", "avg"],
        colors = ["red", "green", "royalblue"],
        show_legend: bool = True,
        legend_params: dict = dict(
            bbox_to_anchor=(0.5, 1.03), loc="lower center", fontsize=10
        ),
    ) -> None:
        """
        Plots the combined statistics development of all runs.

        If runs should be sorted in a specific way, use sort_runs_by_names in init.
        """
        total_x_offset = 0
        counter = 0
        for run_name, run_analysis in self.runs_analysis_dict.items():

            if counter == 0 and show_legend:
                show_legend = True
            else:
                show_legend = False

            run_analysis.plot_combined_statistics_development(
                ax=ax,
                display_stage_id=display_stage_id,
                stage_id_y_pos=stage_id_y_pos,
                stage_id_x_offset=stage_id_x_offset,
                value_types=value_types,
                colors=colors,
                x_offset=total_x_offset,
                show_legend=show_legend,
                legend_params=legend_params
            )
            ax.axvline(
                total_x_offset, color="red", linestyle="-", linewidth=4
            )

            total_x_offset += run_analysis.get_n_generations()

            counter += 1

    def create_statistics_value_plots(
        self,
    ) -> None:
        """
        Saves plots of the config values.
        """
        save_dir = os.path.join(
            self.analysis_results_dir, 
            "statistics_plots", 
        )
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for run_name, run_analysis in self.runs_analysis_dict.items():
            fig, ax = plt.subplots(
                2, 1, figsize=(19, 14), sharex=True,
                height_ratios=[3, 1], tight_layout=True
            )
            fig.suptitle("Similarity fitness development")

            ax_fit = ax[0]
            ax_fit.set_ylim(-0.01, 1.1)
            ax_fit.set_ylabel("Fitness")
            run_analysis.plot_combined_statistics_development(
                ax=ax_fit,
                display_stage_id=True,
                stage_id_y_pos=1.,
                stage_id_x_offset=2.5,
                value_types=["max", "min", "avg"],
                colors=["red", "green", "royalblue"]
            )

            ax_std = ax[1]
            ax_std.set_ylabel("Std")
            ax_std.set_xlabel("Generation")
            run_analysis.plot_combined_statistics_development(
                ax=ax_std,
                display_stage_id=False,
                value_types=["std"],
                colors=["orange"]
            )

            plt.savefig(os.path.join(save_dir, f"{run_name}.png"))
            plt.close()

    def plot_stats_key_hist(
        self,
        ax: Axes,
        key: str | None = None,
        run_was_completed: bool = True,
        color: str = "black",
        alpha: float = 0.5,
        bins: list[float] | np.ndarray = np.arange(0, 1.1, 0.1),
        bin_width: float = 0.09,
        bin_edge_offset: float | int = 0.,
        add_text: bool = True,
        plot_vlines_between_bins: bool = True,
    ) -> None:
        """
        Plots a histogram of the stats key. If key is None, it will plot 
        the key: "best_{self.main_stat_name}".
        """
        if key is None:
            key = f"best_{self.main_stat_name}"

        analysis_results_dict = self.get_analysis_results_dict()

        temp_values = []
        for best_crystal_value, was_comp in zip(
            analysis_results_dict[key],
            analysis_results_dict["was_completed"]
        ):
            if was_comp == run_was_completed:
                temp_values.append(best_crystal_value)

        values = []
        for value in temp_values:
            if value is None:
                continue
            values.append(value)

        if len(values) == 0:
            return

        hist, bin_edge = np.histogram(
            values, bins=bins
        )

        ax.bar(
            bin_edge[:-1] + bin_edge_offset,
            hist,
            width=bin_width,
            color=color,
            alpha=alpha,
            label=self.name
        )

        if add_text:
            text = "Finished" if run_was_completed else "Unfinished"
            bbox_props = dict(
                boxstyle="round", fc="w", ec="0.5", alpha=0.8, color="black"
            )

            ax.text(
                0.5, 0.85, text, ha="center", va="center", 
                transform=ax.transAxes, bbox=bbox_props
            )

        if plot_vlines_between_bins:
            bin_spacing = bin_edge[1] - bin_edge[0]
            ax.axvline(
                bins[0] - bin_spacing/2, color="black", linestyle="--", 
                linewidth=2
            )

            for bin_start in bins:
                ax.axvline(
                    bin_start + bin_spacing/2, color="black", linestyle="--", 
                    linewidth=2
                )

    def plot_best_and_target_crystals(self) -> None:
        save_dir = os.path.join(
            self.analysis_results_dir, "best_and_target_crystals"
        )
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for run_name, run_analysis in self.runs_analysis_dict.items():
            best_crystal_tuple = run_analysis.get_best_crystal_tuple()
            if best_crystal_tuple == None:
                best_crystal_value = 0
            else:
                best_crystal_value = best_crystal_tuple[1]

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(
                f"{run_name} - " + 
                f"{self.main_stat_name}: {best_crystal_value:.3f} - " +
                f"Completed: {run_analysis.was_completed()}",
                fontsize=25
            )
            ax[0].set_xlim(0, 20)
            ax[0].set_ylim(0, 20)
            ax[1].set_xlim(0, 20)
            ax[1].set_ylim(0, 20)

            run_analysis.plot_best_crystal_and_target(
                ax_target=ax[0], ax_best=ax[1]
            )

            ax[0].set_title("Target crystal", fontsize=15)
            ax[1].set_title("Best crystal", fontsize=15)

            save_path = os.path.join(save_dir, f"{run_name}.png")
            plt.savefig(save_path)
            plt.close()

    def plot_target_density_vs_best_density(
        self,
        ax: Axes,
        marker = ".",
        linewidth: int = 2,
        color_complete = "black",
        size_complete = 300,
        color_incomplete = "orange",
        size_incomplete = 100,
    ) -> None:
        runs_analysis_results_dict = self.get_analysis_results_dict()
        colors_list = []
        size_list = []
        for completed_bool in runs_analysis_results_dict["was_completed"]:
            if completed_bool:
                colors_list.append(color_complete)
                size_list.append(size_complete)
            else:
                colors_list.append(color_incomplete)
                size_list.append(size_incomplete)

        ax.scatter(
            runs_analysis_results_dict["density_target"],
            runs_analysis_results_dict["density_best"],
            c=colors_list,
            s=size_list,
            marker=marker,
            label=self.name,
            linewidths=linewidth,
        )

    def plot_n_generations_vs_best_stat(
        self,
        ax: Axes,
        symbol = ".",
        linewidth: int = 2,
        color_complete = "black",
        size_complete = 300,
        color_incomplete = "orange",
        size_incomplete = 100,
    ) -> None:

        runs_analysis_results_dict = self.get_analysis_results_dict()
        colors_list = []
        size_list = []
        for completed_bool in runs_analysis_results_dict["was_completed"]:
            if completed_bool:
                colors_list.append(color_complete)
                size_list.append(size_complete)
            else:
                colors_list.append(color_incomplete)
                size_list.append(size_incomplete)

        ax.scatter(
            runs_analysis_results_dict["n_generations"],
            runs_analysis_results_dict[f"best_{self.main_stat_name}"],
            label=self.name,
            c=colors_list,
            s=size_list,
            marker=symbol,
            linewidths=linewidth,
        )

    def plot_analysis_results_dict_keys(
        self,
        ax: Axes,
        x_key: str,
        y_key: str,
        run_was_completed = True,
        linewidth: int = 2,
        color = "black",
        size = 300,
        marker: str | None = None,
        edgecolor = "gray",
        alpha: float | int = 1,
        zorder: int = 1,
        label: str | None = None,
    ) -> None:
        """
        Plots two parameters against each other from the analysis results dict.
        Example:

            fig, ax = plt.subplots()
            plot_analysis_results_dict_keys(
                ax, "n_generations", "best_similarity"
            )
        """
        x_values = []
        y_values = []
        runs_analysis_results_dict = self.get_analysis_results_dict()
        for i, completed_bool in enumerate(
            runs_analysis_results_dict["was_completed"]
        ):
            if completed_bool == run_was_completed:
                x_values.append(runs_analysis_results_dict[x_key][i])
                y_values.append(runs_analysis_results_dict[y_key][i])

        if label is None:
            label = self.name

        ax.scatter(
            x_values,
            y_values,
            label=label,
            c=color,
            s=size,
            marker=marker,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
            alpha=alpha
        )

# ╔══════════════════════════════════════════════════════════╗
# ║                    Analysis Functions                    ║
# ╚══════════════════════════════════════════════════════════╝

def create_analysis_plot(
    analysis_many_runs: AnalyseManyRuns,
    y_key: str,
    x_key: str,
    y_label: str | None = None,
    x_label: str | None = None,
    size: int = 300,
    linewidth: int = 1,
    marker: str = "o",
    alpha: float | int = .9,
    save_fig: bool = False,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    figsize: tuple[int|float, int|float] | None = None,
    show_legend: bool = True,
    legend_params: dict = dict(
        bbox_to_anchor=(0.5, 1.03), loc="lower center", fontsize=10
    ),
) -> tuple[Figure, Axes]:
    if y_label is None:
        y_label = y_key
    if x_label is None:
        x_label = x_key

    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=figsize, tight_layout=True
    )
    ax: Axes = ax

    ax.set_ylabel(y_label)
    ax.grid(True)
    analysis_many_runs.plot_analysis_results_dict_keys(
        ax=ax,
        y_key=y_key,
        x_key=x_key,
        linewidth=linewidth,
        marker=marker,
        alpha=alpha,
        size=size,
        run_was_completed=True,
        color="black",
        label="Finished",
    )
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    analysis_many_runs.plot_analysis_results_dict_keys(
        ax=ax,
        y_key=y_key,
        x_key=x_key,
        linewidth=linewidth,
        marker=marker,
        alpha=alpha,
        size=size,
        run_was_completed=False,
        color="orange",
        label="Unfinished",
    )

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    if show_legend:
        legend_params["ncol"] = 2
        ax.legend(**legend_params)

    if save_fig:
        save_path = os.path.join(
            analysis_many_runs.analysis_results_dir, 
            f"{y_key}_vs_{x_key}.png"
        )
        plt.savefig(save_path)
        plt.close()

        print(f"Saved plot to {save_path}")
    return fig, ax

def create_combined_statistics_development_plot(
    analysis_many_runs: AnalyseManyRuns,
    display_stage_id: bool = True,
    stage_id_y_pos: float = 1.,
    stage_id_x_offset: float = 2.5,
    value_types = ["max", "min", "avg"],
    colors = ["red", "green", "royalblue"],
    save_fig: bool = False,
    figsize: tuple[int|float, int|float] | None = None,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    show_legend: bool = True,
    legend_params: dict = dict(
        bbox_to_anchor=(0.5, 1.03), loc="lower center", fontsize=10
    ),
) -> tuple[Figure, Axes]:

    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=figsize, tight_layout=True
    )
    ax: Axes = ax

    ax.set_ylabel("Fitness")
    ax.set_xlabel("Generation")
    ax.grid(True)
    analysis_many_runs.plot_combined_statistics_development(
        ax=ax,
        display_stage_id=display_stage_id,
        stage_id_y_pos=stage_id_y_pos,
        stage_id_x_offset=stage_id_x_offset,
        value_types=value_types,
        colors=colors,
        show_legend=show_legend,
        legend_params=legend_params
    )

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # if show_legend:
    #     legend_params["ncol"] = len(value_types)
    #     ax.legend(**legend_params)

    if save_fig:
        save_path = os.path.join(
            analysis_many_runs.analysis_results_dir, 
            "combined_statistics_development.png"
        )
        plt.savefig(save_path)
        plt.close()

        print(f"Saved plot to {save_path}")
    return fig, ax


# ╔══════════════════════════════════════════════════════════╗
# ║                      Example Usage                       ║
# ╚══════════════════════════════════════════════════════════╝

def main(folder_path: str) -> None:
    cellbounds = CustomCellBounds({"a": [1, 16], "b": [1, 16], "c": [1, 16]})

    many_runs_results = ManyRunsResults(folder_path)
    analysis = AnalyseManyRuns(
        many_runs_results, 
        cell_bounds=cellbounds,
        min_num_stages=23,
        main_stats_key="soap_similarity_strong_RBFSimilarity",
        main_stat_name="similarity",
        sort_runs_by_names=[f"run_{i}" for i in range(7, 21)],
    )

    print(f"Possible keys: {analysis.get_analysis_results_dict_keys()}")

    overview = analysis.get_analysis_overview_str(round_values=4)
    results_table = analysis.get_analysis_results_table(round_values=4)


    # ╓                                                          ╖
    # ║                Plot global run statistics                ║
    # ╙                                                          ╜

    # create_analysis_plot(
    #     analysis, 
    #     x_key=f"best_{analysis.main_stat_name}",
    #     y_key="n_generations",
    #     y_label="Similarity", 
    #     x_label="Number of generations",
    #     save_fig=True,
    #     x_lim=(0, 100),
    #     y_lim=(0, 1),
    # )

    create_combined_statistics_development_plot(
        analysis,
        display_stage_id=False,
        value_types=["max", "min", "avg"],
        colors=["red", "green", "royalblue"],
        save_fig=False,
        figsize=(19, 14),
    )
    plt.show()


    # ╓                                                          ╖
    # ║                   Plot run statistics                    ║
    # ╙                                                          ╜
    # analysis.plot_best_and_target_crystals()
    # analysis.create_statistics_value_plots()

    analysis_overview_path = os.path.join(
        analysis.analysis_results_dir, "analysis_overview.txt"
    )
    with open(analysis_overview_path, "w") as file:
        file.write(overview)
        file.write("\n\n")
        if isinstance(results_table, str):
            file.write(results_table)

    print("Analysis results:")
    print(overview)
    print(results_table)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} path/to/run/dir")
        sys.exit(1)

    folder_path = sys.argv[1]

    main(folder_path)
