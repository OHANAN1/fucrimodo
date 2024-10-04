from operator import xor
from typing import Any, NoReturn

from numpy.typing import NDArray
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.analysis.results_classes import RunResults, StageResults
from ase.db.core import Database
import numpy as np
import ase
from collections import Counter
import warnings
import os
from ase.visualize.plot import plot_atoms
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from icecream import ic
import pandas as pd

# ╔══════════════════════════════════════════════════════════╗
# ║                    Utility functions                     ║
# ╚══════════════════════════════════════════════════════════╝

def check_if_crystals_have_same_stoichiometry(
    crystal1: ase.Atoms,
    crystal2: ase.Atoms
) -> tuple[bool, float | None]:
    """
    Checks if two crystals have the same composition.
    :returns: A tuple:

        - bool: crystal 1 and 2 have the same stoichiometry
        - float: ratio of elements, or None if not the same stoichiometry
    """
    cry1_atomic_num = crystal1.get_atomic_numbers()
    cry2_atomic_num = crystal2.get_atomic_numbers()

    for atomic_num in cry1_atomic_num:
        if atomic_num not in cry2_atomic_num:
            return False, None

    for atomic_num in cry2_atomic_num:
        if atomic_num not in cry1_atomic_num:
            return False, None

    counter1 = Counter(cry1_atomic_num)
    counter2 = Counter(cry2_atomic_num)

    elements1, counts1 = zip(*counter1.items())
    elements2, counts2 = zip(*counter2.items())

    ratios = [
        count2 / count1
        for count1, count2 in zip(sorted(counts1), sorted(counts2))
    ]

    ratios = [
        count2 / count1
        for count1, count2 in zip(sorted(counts1), sorted(counts2))
    ]

    if all(ratio == ratios[0] for ratio in ratios):
        return True, ratios[0]
    else:
        return False, None


# ╔══════════════════════════════════════════════════════════╗
# ║                       Main classes                       ║
# ╚══════════════════════════════════════════════════════════╝

class AnalyseStage():
    """
    Object to loads and analysis the :class:`StageResults` class.
    Best used in combination with :class:`AnalyseRun`, since it automatically 
    loads every AnalyseStage class.

    :param stage_results: If stage results are not already present, give the
        run dictionary and stage id as tuple.

    :raises KeyError: If a selected main_stats_key is not available.
    """
    def __init__(
        self,
        stage_results: StageResults | tuple[str, int],
        cell_bounds: CustomCellBounds | None = None,
    ) -> None:
        self.stage_results = stage_results
        self.cell_bounds = cell_bounds

    @property
    def stage_results(self) -> StageResults:
        return self._stage_results

    @stage_results.setter
    def stage_results(self, value: StageResults | tuple[str, int] ) -> None:
        """Loads the stage results class.
        
        If set with a :class:`StageResults` object, the object is directly
        used. If set with a tuple, the run directory and stage id are used to
        load the stage results.
        """
        if type(value) == tuple[str, int]:
            self._stage_results = StageResults(
                run_dir=value[0], id=value[1]
            )
        elif type(value) == StageResults:
            self._stage_results = value
        else:
            raise ValueError(
                f"Expected tuple or StageResults, got {type(value)}"
            )

    @property
    def analysis_types(self) -> list[str]:
        """Returns a list of all analysis types that can be performed.

        Possible types depend on the type of stage that is analysed.
        This is determined by the str in :attr:`StageResults.info_dict["type"]`.
        For example if a GAStage is analysed, the possible types are:

            - "fitness"
            - "crossovers"
            - "mutations"
        """
        # Load up which type is saved in the info_dict
        determinded_type = self.stage_results.info_dict["type"]

        # Return the possible analysis types depending on the stage type
        if determinded_type == "GAStage":
            return ["fitness", "crossover", "mutation"]
        else:
            raise NotImplementedError(
                "Only GAStage is implemented so far. But the stage type found "
                f"in the :attr:`StageResults.info_dict` is {determinded_type}"
            )

    def plot_results(
        self, 
        analysis_type: str,
        ax: Axes, 
        row: int,
        x_key: str = "gen",
        x_label: str = "Generation",
        y_keys: list[str] | None = None,
        y_label: str = ""
    ) -> None:
        """Plots the data of the results of a crossover operator.

        :param analysis_type: Type of analysis that should be performed.
            Possible are stored in :attr:`analysis_types` and depend on the
            type of stage that is analysed.
        :param ax: Matplotlib axis object to plot on.
        :param row: Index of the row in the crossover results dataframe of
            which the results data should be plotted.
        :param x_key: Key of the x-axis data. Normally the generation is used
            as x-axis, with the key "gen".
        :param y_keys: List of keys of the y-axis data that should be plotted.
            If None, all columns of the results dataframe are plotted.
            To get the column names use
            :code:`self.stage_results.crossovers.loc[row, "results"].columns`.

        :raises IndexError: If the given crossover index is out of range.
        """
        # Load name and results_df depending on the analysis type
        if analysis_type == "fitness":
            name = self.stage_results.fitnesses.at[row, "names"]
            results_df: pd.DataFrame = self.stage_results.fitnesses.loc[
                row, "results"
            ]

        elif analysis_type == "mutation":
            name = self.stage_results.mutations.at[row, "names"]
            results_df: pd.DataFrame = self.stage_results.mutations.loc[
                row, "results"
            ]

        elif analysis_type == "crossover":
            name = self.stage_results.crossovers.at[row, "names"]
            results_df: pd.DataFrame = self.stage_results.crossovers.loc[
                row, "results"
            ]

        else:
            raise ValueError(
                "The given analysis type is not valid for this Stage.\n"
                f"Possible types: {self.analysis_types}"
            )

        # Plot the selected results data
        results_df.plot(
            ax=ax,
            x=x_key,
            y=y_keys,
            title=f"{analysis_type}: {name}"
        )

        # Set labels of the plot
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    def get_overview_table(self, analysis_type: str) -> str:
        """Creates an overview table of the operators with their index, names
        and representations.

        :param analysis_type: Type of analysis that should be performed.
            Possible are stored in :attr:`analysis_types` and depend on the
            type of stage that is analysed.

        :raises ValueError: If the given analysis type is not valid.
        """
        if analysis_type == "fitness":
            info_df = pd.DataFrame(
                self.stage_results.fitnesses, 
                columns=["names", "reprs"] # type: ignore
            )
        elif analysis_type == "mutation":
            info_df = pd.DataFrame(
                self.stage_results.mutations, 
                columns=["names", "reprs"] # type: ignore
            )
        elif analysis_type == "crossover":
            info_df = pd.DataFrame(
                self.stage_results.crossovers, 
                columns=["names", "reprs"] # type: ignore
            )
        else:
            raise ValueError(
                "The given analysis type is not valid for this Stage.\n"
                f"Possible types: {self.analysis_types}"
            )

        # Return the overview table as string
        return str(info_df)

    def get_best_crystal_tuple(
        self, statistics_key: str, invert: bool = False
    ) -> tuple[ase.Atoms, float, dict[str, Any]]:
        """
        Returns the crystal with the highest value for the given 
        :data:`statistics key` that was found in the crystals database
        key value pairs.

        :param statistics_key: Key from the crystals database with the desired
            statistic.
        :param invert: If False the sorting will be inverted, meaning the 
            crystal with the lowest value of the desired statistic is returned

        :return:
            - best crystal
            - statistics value of crystal for the given statistics key
            - key value pairs of the best crystal
        """
        statistic_values = []
        for key_value_pair in self.stage_results.key_value_pairs:
            statistic_values.append(key_value_pair[statistics_key])

        if invert == False:
            crystal_index = np.argmax(statistic_values)
        else: 
            crystal_index = np.argmin(statistic_values)

        return (
            self.stage_results.crystals[crystal_index],
            self.stage_results.key_value_pairs[crystal_index][statistics_key],
            self.stage_results.key_value_pairs[crystal_index]
        )


class AnalyseRun():
    """ 
    Class to analyse the results that where collected during the run.

    :param run_results: Either the :class:`RunResults` object that should be 
        analyzed or the path to the run directory.

    :raises FileNotFoundError: If expected files are missing.
    """
    def __init__(
        self, 
        run_results: RunResults | str,
    ) -> None:
        self.run_results = run_results
        self.stages = self.run_results

    @property
    def run_results(self) -> RunResults:
        """
        :class:`RunResults` object that collects and sorts all data that was 
        saved during a run.
        Can be set with the :class:`RunResults` object or the path to the run
        directory that should be loaded.
        """
        return self._run_results

    @run_results.setter
    def run_results(self, value: RunResults | str):
        if type(value) == str:
            self._run_results = RunResults(run_dir=value)
        elif type(value) == RunResults:
            self._run_results = value

    @property
    def stages(self) -> list[AnalyseStage]:
        """
        Ordered list of :class:`AnalyseStage` objects for each 
        stage in the run.
        """
        return self._stages

    @stages.setter
    def stages(self, value: RunResults):
        self._stages = []
        for stage in value.stages:
            self._stages.append(AnalyseStage(stage_results=stage))

    def get_global_statistics_keys(self) -> list[str]:
        """List of the global statistic keys that all stages share."""
        return list(self.run_results.global_statistics_log.chapters.keys())

    def get_number_of_stages(self) -> int:
        return self.run_results.n_stages

    def get_best_crystal_tuple(
        self, 
        statistics_key: str, 
        invert: bool = False
    ) -> tuple[ase.Atoms, float, dict[str, Any], int]:
        """
        Returns the crystal with the highest value for the given 
        :data:`statistics key` that was found in the crystals database
        key value pairs.

        :param statistics_key: Key from the crystals database with the desired
            statistic.
        :param invert: If False the sorting will be inverted, meaning the 
            crystal with the lowest value of the desired statistic is returned.

        :return:
            - best crystal
            - statistics value of crystal for the given statistics key
            - key value pairs of the best crystal
            - id of the stage the crystal is from
        """
        crystal_tuples = []
        stat_values = []
        for stage_analys in self.stages:
            crystal, stat_val, key_value_pair = stage_analys.get_best_crystal_tuple(
                statistics_key=statistics_key, invert=invert
            )
            crystal_tuples.append(
                (
                    crystal, 
                    stat_val, 
                    key_value_pair, 
                    stage_analys.stage_results.id
                )
            )
            stat_values.append(stat_val)

        if invert:
            best_index = np.argmin(stat_values)
        else:
            best_index = np.argmax(stat_values)

        return crystal_tuples[best_index]

    def get_n_generations(self) -> int:
        """Returns the total number of steps of the whole run."""
        total_steps = 0
        for stage_analys in self.stages:
            total_steps += stage_analys.stage_results.n_generations
        return total_steps

    # def get_global_statistic_values_all_stages(
    #     self,
    #     statistics_key: str, 
    #     value_type: str | None = None
    # ) -> dict[str, dict[str, list]] | dict[str, list]:
    #     """Returns all found values for a specific statistic that was collected
    #     during all stages. The desired statistic must be present in all 
    #     stages.
    #
    #     :param statistics_key: Key of the statistic of interest. Must be 
    #         present in all stages.
    #     :param value_type: If not None, only the specified value type is
    #         returned. Normally these types are: "max", "min", "avg" or "std".
    #
    #     :returns: A dict with with keys :attr:`AnalyseStage.id` as keys and the 
    #         data associated with the given :data:`statistics_key` of the 
    #         specific stage as value.
    #         That is either a dict with keys 'max', 'min', 'avg' and 'std' for the 
    #         selected statistic. Or only the list of values for the value type
    #         that is specified.
    #
    #     :raises KeyError: If the given :data:`statistics_key` could not be 
    #         found in all stages.
    #     """
    #     assert statistics_key in self.get_global_statistics_keys(), (
    #         f"Key {statistics_key} not in shared keys."
    #     )
    #
    #     statistics_values = {}
    #     for stage_analys in self.stages:
    #         stage_id = stage_analys.id
    #         values_dict = stage_analys.get_global_statistics_values(
    #             statistics_key, value_type
    #         )
    #         statistics_values[stage_id] = values_dict
    #
    #     return statistics_values

    def get_analysis_results_dict(
        self,
        statistics_key: str,
        target_crystal: ase.Atoms | None = None,
        cellbounds: CustomCellBounds | None = None,
        round_values: int = 3,
    ) -> dict:
        """Generates a dictionary with all kinds of statistics data.

        :returns: Dictionary of analysed data with keys:
            - run_name
            - n_generations: Total number of generations.
            - best_crystal: Crystal with the highest value for provided statistics key.
            - best_crystal_value: The value for the providid statistics key.
            - best_crystal_key_value_pairs: Key value pairs of the best crystal.
            - best_crystal_density: Density of the best crystal.
            - best_crystal_volume: Volume of the best crystal.
            - best_crystal_n_atoms: Number of atoms of the best crystal.
            - found_in_stage: Stage id the best crystal was found in.
            if target crystal is provided:
            - target_crystal
            - target_crystal_n_atoms: Number of atoms of target crystal.
            - target_crystal_volume: Volume of target crystal.
            - same_stoichiometry: True if stoichiometry of target and best found crystal is the same.
            - ratio: Ratio of atom types of best found crystals and target.
            - same_composition: True if composition of target and best found crystal is the same.
        """
        best_crystal_tuple = self.get_best_crystal_tuple(
            statistics_key=statistics_key
        )
        best_cry, best_cry_value, best_cry_key_val_pair, stage_id = best_crystal_tuple
        if isinstance(best_cry_value, float):
            best_cry_value = round(best_cry_value, round_values)

        analysis_results_dict = {}
        analysis_results_dict["run_name"] = self.run_results.run_name
        analysis_results_dict["n_generations"] = self.get_n_generations()

        analysis_results_dict["best_crystal"] = best_cry
        analysis_results_dict["best_crystal_key_value_pairs"] = best_cry_key_val_pair 
        analysis_results_dict["best_crystal_value"] = best_cry_value
        analysis_results_dict["best_crystal_density"] = len(best_cry) / best_cry.get_volume()
        analysis_results_dict["best_crystal_volume"] = best_cry.get_volume()
        analysis_results_dict["best_crystal_n_atoms"] = len(best_cry)
        analysis_results_dict["found_in_stage"] = stage_id

        if target_crystal is not None:
            same_stoichi, ratio = check_if_crystals_have_same_stoichiometry(
                best_cry, target_crystal
            )
            same_comp=False
            if (same_stoichi == True) and ratio == 1: 
                same_comp = True

            analysis_results_dict["target_crystal"] = target_crystal
            analysis_results_dict["target_crystal_n_atoms"] = len(target_crystal)
            analysis_results_dict["target_crystal_volume"] = target_crystal.get_volume()

            analysis_results_dict["same_stoichiometry"] = same_stoichi
            analysis_results_dict["ratio"] = ratio
            analysis_results_dict["same_composition"] = same_comp

            if cellbounds is not None:
                analysis_results_dict["target_in_bounds"] = cellbounds.is_within_bounds(
                    cell=target_crystal.cell
                )

        return analysis_results_dict

    # def plot_combined_statistics_development(
    #     self,
    #     ax: Axes,
    #     statistics_key: str,
    #     display_stage_id: bool = True,
    #     stage_id_y_pos: float = 1.,
    #     stage_id_x_offset: float = 2.5,
    #     value_types: list[str] = ["max", "min", "avg"],
    #     colors: list[str] = ["red", "green", "royalblue"],
    #     x_offset: int = 0,
    #     show_legend: bool = True,
    #     legend_params: dict = dict(
    #         bbox_to_anchor=(0.5, 1.03), loc="lower center", fontsize=25
    #     ),
    # ) -> None:
    #     """
    #     Plots the main statistics values for the whole run.
    #     Use value_types to specify which values to plot.
    #     Normally ("max", "min", "avg") and "std" are used.
    #     Give the colors for the value types in the same order.
    #
    #     Use x_offset to change start position of the x-axis.
    #     Can be used to plot multiple runs in the same plot.
    #     NOTE: support for floats will be added later
    #     """
    #     assert len(value_types) == len(colors), (
    #         "Need the same amount of colors as value types."
    #     )
    #     stage_bbox = dict(
    #         boxstyle="square,pad=0.15", facecolor='white', alpha=1., 
    #         edgecolor='black', linewidth=2.
    #     )
    #
    #     statistics_values = self.get_global_statistic_values_all_stages(
    #         statistics_key
    #     )
    #
    #     x_start = x_offset
    #     for stage_id, stat_val_dict in statistics_values.items():
    #         if not isinstance(stat_val_dict, dict):
    #             raise ValueError(
    #                 f"Expected dict, got {type(stat_val_dict)}."
    #             )
    #
    #         assert all(
    #             value_type in stat_val_dict.keys()
    #             for value_type in value_types
    #         ), f"Value types {value_types} not in statistics values."
    #
    #         total_len = len(stat_val_dict[value_types[0]])
    #
    #         x_end = x_start + total_len
    #         x = range(x_start, x_end)
    #
    #         if display_stage_id:
    #             ax.text(
    #                 x_start + stage_id_x_offset,
    #                 stage_id_y_pos,
    #                 str(stage_id),
    #                 bbox=stage_bbox,
    #             )
    #
    #         # only add lable once
    #         if stage_id == 1:
    #             for value_type, color in zip(value_types, colors):
    #                 ax.plot(
    #                     x, stat_val_dict[value_type], color=color, 
    #                     label=value_type
    #                 )
    #         else:
    #             for value_type, color in zip(value_types, colors):
    #                 ax.plot(x, stat_val_dict[value_type], color=color)
    #
    #         x_start = x_end - 1
    #         ax.axvline(x=x_start, color="black", linestyle="-", linewidth=2.)
    #
    #     ax.set_xlim(0, x_start)
    #
    #     if show_legend:
    #         legend_params["ncol"] = len(value_types)
    #         ax.legend(**legend_params)

    def plot_best_crystal_and_target(
        self, 
        ax_target: Axes, 
        ax_best: Axes, 
        target_crystal: ase.Atoms,
        statistics_key: str,
    ) -> None:
        """Plots the best crystal and the target crystal of a run."""
        plot_atoms(
            atoms=target_crystal,
            ax=ax_target,
            radii=0.5,
            rotation="20x,20y,0z"
        )

        best_crystal_tuple = self.get_best_crystal_tuple(statistics_key)
        if best_crystal_tuple is not None:
            plot_atoms(
                atoms=best_crystal_tuple[0],
                ax=ax_best,
                radii=0.5,
                rotation="20x,20y,0z"
            )

        ax_target.set_axis_off()
        ax_best.set_axis_off()

    # def get_global_statistics_values(
    #     self, statistics_key: str, value_type: str | None = None
    # ) -> dict[str, list] | list:
    #     """
    #     Returns all found values for a specific statistic that was collected
    #     during the stage.
    #
    #     :param statistics_key: Key of the statistic of interest
    #     :param value_type: If not None, only the specified value type is
    #         returned. Normally these types are: "max", "min", "avg" or "std".
    #
    #     :returns: The data associated with the given :data:`statistics_key`. 
    #         Either a dict with keys 'max', 'min', 'avg' and 'std' for the 
    #         selected statistic. Or only the list of values for the value type 
    #         that is specified.
    #
    #     :raises KeyError: If the given :data:`statistics_key` could not be 
    #         found in the data saved during the stage.
    #     """
    #     self.__is_valid_global_statics_keys(
    #         key=statistics_key, raise_error=True
    #     )
    #     stat_values = self._stage_results.global_statistics_log[statistics_key]
    #
    #     if value_type is None:
    #         return stat_values
    #     else:
    #         return stat_values[value_type]

    # def __is_valid_global_statics_keys(
    #     self, key: str | None, raise_error: bool = False
    # ) -> NoReturn | bool :
    #     """
    #     Checks if the provided key is found in the keys 
    #     of :attr:`StageResults.stage_dict`.
    #     None values are also counted as invalid.
    #
    #     :param key: Key that should be checked.
    #     :param raise_error: If True will raise error if the key is not valid.
    #
    #     :raise KeyError: Only when :data:raise_error is True and provided key
    #         is not valid.
    #     """
    #     valid_keys = self._stage_results.global_statistics_log.keys()
    #     if key in valid_keys:
    #         return True
    #     else:
    #         if raise_error:
    #             raise KeyError(
    #                 f"Provided key {key} was not found in valid keys:\n" 
    #                 f"{valid_keys}"
    #             )
    #         else:
    #             return False


# ╔══════════════════════════════════════════════════════════╗
# ║                    Analysis Functions                    ║
# ╚══════════════════════════════════════════════════════════╝

def create_combined_statistics_development_plot(
    run_analysis: AnalyseRun,
    statistics_key: str,
    display_stage_id: bool = True,
    stage_id_y_pos: float = 1.,
    stage_id_x_offset: float = 2.5,
    statistics_name: str | None = None,
    statistics_symbol: str | None = None,
    x_lim: tuple[int, int] | None = None,
    y_lim: tuple[float, float] | None = None,
    save_fig: bool = False,
    legend_params: dict = dict(
        bbox_to_anchor=(0.5, 1.03), loc="lower center", fontsize=25
    ),
    y_scale: str = "linear",
    height_ratios: list[float] = [2, 1],
    fig_size: tuple[int, int] = (18, 11),
    analysis_dir_path: str | None = None
) -> tuple[Figure, np.ndarray[Axes,Any]]:
    """
    Plots the main statistics values for the whole run.
    Use value_types to specify which values to plot.
    Normally ("max", "min", "avg") and "std" are used.
    Give the colors for the value types in the same order.

    If the statistics name is none the main statistics name or key will be 
    used as the y-axis label.
    The statistics symbol is added to the y-axis label and used in the
    std plot. If none, the first letter of the statistics name is used.
    """
    
    fig, ax = plt.subplots(
        2, 1, figsize=fig_size, sharex=True,
        height_ratios=height_ratios, tight_layout=True
    )
    assert type(ax) == np.ndarray, "Somehow axes where initialized falsely"

    ax_fit = ax[0]
    if y_lim is not None: ax_fit.set_ylim(y_lim)
    if x_lim is not None: ax_fit.set_xlim(x_lim)

    ax_fit.set_yscale(y_scale)

    if statistics_name is None:
        statistics_name = statistics_key

    if statistics_symbol is None:
        statistics_symbol = statistics_name[0]

    ax_fit.set_ylabel(f"{statistics_name} {statistics_symbol}")
    run_analysis.plot_combined_statistics_development(
        ax=ax_fit,
        display_stage_id=display_stage_id,
        stage_id_y_pos=stage_id_y_pos,
        stage_id_x_offset=stage_id_x_offset,
        value_types=["max", "min", "avg"],
        colors=["red", "green", "royalblue"],
        statistics_key=statistics_key,
        show_legend=True,
        legend_params=legend_params
    )

    ax_std = ax[1]
    ax_std.set_ylabel(f"$\\sigma$({statistics_symbol})")
    ax_std.set_xlabel("Generation")
    run_analysis.plot_combined_statistics_development(
        ax=ax_std,
        display_stage_id=False,
        value_types=["std"],
        colors=["orange"],
        statistics_key=statistics_key,
        show_legend=False
    )

    if save_fig:
        save_name = f"{statistics_key}_development.png"
        if analysis_dir_path is not None:
            fig.savefig(os.path.join(analysis_dir_path, save_name))
        else:
            fig.savefig(save_name)

    return fig, ax

if __name__ == "__main__":

    import sys

    try:
        run_dir = sys.argv[1]
    except IndexError:
        print("Please use as: python path/to/script.py path/to/run_dir")
        sys.exit(1)

    if not os.path.exists(run_dir):
        print("Path does not exist")
        sys.exit(1)

    run = StageResults(run_dir, 1)
    analysis = AnalyseStage(run)

    print(analysis.get_overview_table(analysis_type="fitness"))
    print()
    print(analysis.get_overview_table(analysis_type="mutation"))
    print()
    print(analysis.get_overview_table(analysis_type="crossover"))

    fig, axes = plt.subplots(1, 1)
    analysis.plot_results(
        analysis_type="mutation",
        ax=axes,
        row=0,
        y_label="Number of events"
    )

    fig, axes = plt.subplots(1, 1)
    analysis.plot_results(
        analysis_type="fitness",
        ax=axes,
        row=0,
        y_label="fitness"
    )

    plt.show()
