from typing import Any, NoReturn

from numpy.typing import NDArray
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.analysis.results_class import RunResults, StageResults
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

# ╔══════════════════════════════════════════════════════════╗
# ║                    Utility functions                     ║
# ╚══════════════════════════════════════════════════════════╝

def check_if_crystals_have_same_stoichiometry(
    crystal1: ase.Atoms,
    crystal2: ase.Atoms
) -> tuple[bool, float | None]:
    """
    Checks if two crystals have the same composition.
    Returns a tuple:
    :param same_composition: bool
    :param ratio: float
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
    def stage_name(self) -> str:
        return self.stage_results.name

    @property
    def id(self) -> int:
        return self.stage_results.id

    @property
    def stage_results(self) -> StageResults:
        return self._stage_results

    @stage_results.setter
    def stage_results(self, value: StageResults | tuple[str, int] ) -> None:
        if type(value) == tuple[str, int]:
            self._stage_results = StageResults(
                run_dir=value[0], id=value[1]
            )
        elif type(value) == StageResults:
            self._stage_results = value

    @property
    def valid_statistics_keys(self) -> list[str]:
        return list(self._stage_results.stage_dict.keys())

    def get_statistics_values(
        self, statistics_key: str, value_type: str | None = None
    ) -> dict[str, list] | list:
        """
        Returns all found values for a specific statistic that was collected
        during the stage.

        :param statistics_key: Key of the statistic of interest
        :param value_type: If not None, only the specified value type is
            returned. Normally these types are: "max", "min", "avg" or "std".

        :returns: The data associated with the given :data:`statistics_key`. 
            Either a dict with keys 'max', 'min', 'avg' and 'std' for the 
            selected statistic. Or only the list of values for the value type 
            that is specified.

        :raises KeyError: If the given :data:`statistics_key` could not be 
            found in the data saved during the stage.
        """
        self.__is_valid_statics_keys(key=statistics_key, raise_error=True)
        if value_type is not None:
            stat_values = self._stage_results.stage_dict[statistics_key][value_type]
        else:
            stat_values = self._stage_results.stage_dict[statistics_key]

        return stat_values

    def get_best_crystal_tuple(
        self, statistics_key: str, invert: bool = False
    ) -> tuple[ase.Atoms, float]:
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
            self.stage_results.key_value_pairs[crystal_index][statistics_key]
        )

    def __is_valid_statics_keys(
        self, key: str | None, raise_error: bool = False
    ) -> NoReturn | bool :
        """
        Checks if the provided key is found in the keys 
        of :attr:`StageResults.stage_dict`.
        None values are also counted as invalid.

        :param key: Key that should be checked.
        :param raise_error: If True will raise error if the key is not valid.

        :raise KeyError: Only when :data:raise_error is True and provided key
            is not valid.
        """
        valid_keys = self.valid_statistics_keys
        if key in valid_keys:
            return True
        else:
            if raise_error:
                raise KeyError(
                    f"Provided key {key} was not found in valid keys:\n" 
                    f"{valid_keys}"
                )
            else:
                return False


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
        self.stages

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

    def get_shared_statistic_keys(self) -> list[str]:
        """List of the statistic keys that all stages share."""
        all_keys = []
        for stage in self.stages:
            all_keys.append(set(stage.stage_results.stage_dict.keys()))
        all_keys = set.intersection(*all_keys)
        return list(all_keys)

    def get_number_of_stages(self) -> int:
        return self.run_results.n_stages

    def get_best_crystal_tuple(
        self, 
        statistics_key: str, 
    ) -> tuple[ase.Atoms, float, int]:
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
            - key value pairs of the best crystal
            - id of the stage the crystal is from
        """
        crystal_tuples = []
        for stage_analys in self.stages:
            crystal, stat_value = stage_analys.get_best_crystal_tuple(
                statistics_key=statistics_key
            )
            stage_id = stage_analys.id
            crystal_tuples.append((crystal, stat_value, stage_id))

        best_crystal_index = np.argmax(
            [stat_value for _, stat_value, _ in crystal_tuples]
        )
        crystal_tuple = crystal_tuples[best_crystal_index]

        return crystal_tuple

    def get_statistic_values_all_stages(
        self, 
        statistics_key: str, 
        value_type: str | None = None
    ) -> dict[str, dict[str, list]] | dict[str, list]:
        """
        Returns all found values for a specific statistic that was collected
        during all stages. The desired statistic must be present in all 
        stages.

        :param statistics_key: Key of the statistic of interest. Must be 
            present in all stages
        :param value_type: If not None, only the specified value type is
            returned. Normally these types are: "max", "min", "avg" or "std".

        :returns: The data associated with the given :data:`statistics_key`. 
            Either a dict with keys 'max', 'min', 'avg' and 'std' for the 
            selected statistic. Or only the list of values for the value type 
            that is specified.

        :raises KeyError: If the given :data:`statistics_key` could not be 
            found in all stages.
        """
        assert statistics_key in self.get_shared_statistic_keys(), (
            f"Key {statistics_key} not in shared keys."
        )

        statistics_values = {}
        for stage_analys in self.stages:
            stage_id = stage_analys.id
            values_dict = stage_analys.get_statistics_values(
                statistics_key, value_type
            )
            statistics_values[stage_id] = values_dict

        return statistics_values

    # Stopped here
    def get_analysis_results_dict(
        self,
        statistics_key: str,
        statistics_name: str | None = None,
        target_crystal: ase.Atoms | None = None,
        cellbounds: CustomCellBounds | None = None,
        round_values: int = 3,
    ) -> dict:
        """
        Generates a dictionary with all kinds of statistics data.
        :returns: Dictionary with analysed data with keys:
            - run_name
        """
        if statistics_name is None:
            statistics_name = statistics_key

        best_crystal_tuple = self.get_best_crystal_tuple(
            statistics_key=statistics_key
        )

        best_crystal, best_crystal_value, stage_id = best_crystal_tuple

        if isinstance(best_crystal_value, float):
            best_crystal_value = round(best_crystal_value, round_values)

        analysis_results_dict = {}
        analysis_results_dict["run_name"] = self.run_results.run_name
        analysis_results_dict["best_crystal"] = best_crystal
        analysis_results_dict["density_best"] = len(best_crystal) / best_crystal.get_volume()
        analysis_results_dict[f"best_{statistics_key}"] = best_crystal_value
        analysis_results_dict["found_in_stage"] = stage_id
        analysis_results_dict["n_generations"] = self.get_n_generations()
        analysis_results_dict["volume_best"] = best_crystal.get_volume()
        analysis_results_dict["n_atoms_best"] = len(best_crystal)

        if target_crystal is not None:
            analysis_results_dict["target_crystal"] 
            analysis_results_dict["same_composition"] = same_comp
            analysis_results_dict["same_stoichiometry"] = same_stoichi
            analysis_results_dict["ratio"] = ratio
            analysis_results_dict["n_atoms_target"] = len(target_crystal)
            analysis_results_dict["volume_target"] = target_crystal.get_volume()

        n_gen_each_stage = self.get_n_generations_each_stage()
        best_value_each_stage = self.get_best_value_each_stage()
        for stage_id in range(1, self.min_num_stages + 1):
            if stage_id in n_gen_each_stage.keys():
                n_gens = n_gen_each_stage[stage_id]
            else:
                n_gens = None

            analysis_results_dict[f"n_gen_stage_{stage_id}"] = n_gens

            if stage_id in best_value_each_stage.keys():
                best_value = best_value_each_stage[stage_id]
                if isinstance(best_value, float):
                    best_value = round(best_value, round_values)
            else:
                best_value = None

            analysis_results_dict[f"best_value_stage_{stage_id}"] = best_value

        return analysis_results_dict

    def plot_combined_statistics_development(
        self,
        ax: Axes,
        display_stage_id: bool = True,
        stage_id_y_pos: float = 1.,
        stage_id_x_offset: float = 2.5,
        value_types: list[str] = ["max", "min", "avg"],
        colors: list[str] = ["red", "green", "royalblue"],
        statistics_key: str | None = None,
        x_offset: int = 0,
        show_legend: bool = True,
        legend_params: dict = dict(
            bbox_to_anchor=(0.5, 1.03), loc="lower center", fontsize=25
        ),
    ) -> None:
        """
        Plots the main statistics values for the whole run.
        Use value_types to specify which values to plot.
        Normally ("max", "min", "avg") and "std" are used.
        Give the colors for the value types in the same order.

        Use x_offset to change start position of the x-axis.
        Can be used to plot multiple runs in the same plot.
        NOTE: support for floats will be added later
        """
        assert len(value_types) == len(colors), (
            "Need the same amount of colors as value types."
        )
        stage_bbox = dict(
            boxstyle="square,pad=0.15", facecolor='white', alpha=1., 
            edgecolor='black', linewidth=2.
        )

        if self.get_number_of_stages() == 0:
            warnings.warn("Cannot plot, no stages found.")
            return

        statistics_values = self.get_statistic_values_all_stages(
            statistics_key
        )

        x_start = x_offset
        for stage_id, stat_val_dict in statistics_values.items():
            if not isinstance(stat_val_dict, dict):
                raise ValueError(
                    f"Expected dict, got {type(stat_val_dict)}."
                )

            assert all(
                value_type in stat_val_dict.keys()
                for value_type in value_types
            ), f"Value types {value_types} not in statistics values."

            total_len = len(stat_val_dict[value_types[0]])

            x_end = x_start + total_len
            x = range(x_start, x_end)

            if display_stage_id:
                ax.text(
                    x_start + stage_id_x_offset,
                    stage_id_y_pos,
                    str(stage_id),
                    bbox=stage_bbox,
                )

            # only add lable once
            if stage_id == 1:
                for value_type, color in zip(value_types, colors):
                    ax.plot(
                        x, stat_val_dict[value_type], color=color, 
                        label=value_type
                    )
            else:
                for value_type, color in zip(value_types, colors):
                    ax.plot(x, stat_val_dict[value_type], color=color)

            x_start = x_end - 1
            ax.axvline(x=x_start, color="black", linestyle="-", linewidth=2.)

        ax.set_xlim(0, x_start)

        if show_legend:
            legend_params["ncol"] = len(value_types)
            ax.legend(**legend_params)


    def plot_best_crystal_and_target(self, ax_target, ax_best) -> None:
        """
        Plots the best crystal and the target crystal of a run.
        """
        plot_atoms(
            atoms=self.target_crystal,
            ax=ax_target,
            radii=0.5,
            rotation="20x,20y,0z"
        )

        best_crystal_tuple = self.get_best_crystal_tuple(self.main_stats_key)
        if best_crystal_tuple is not None:
            plot_atoms(
                atoms=best_crystal_tuple[0],
                ax=ax_best,
                radii=0.5,
                rotation="20x,20y,0z"
            )

        ax_target.set_axis_off()
        ax_best.set_axis_off()


    def get_n_generations(self) -> int:
        """
        Returns the total number of steps of the whole run.
        """
        total_steps = 0
        for stage_analys in self.stage_analysis_objects:
            total_steps += stage_analys.get_n_generations()
        return total_steps

    def get_n_generations_each_stage(self) -> dict[int, int]:
        """
        Returns the number of generations for each stage.
        """
        n_generations = {}
        for stage_analys in self.stage_analysis_objects:
            n_generations[stage_analys.stage_id] = stage_analys.get_n_generations()
        return n_generations

    def get_best_value_each_stage(self) -> dict[int, float]:
        """
        Returns the highest similarity value for each stage.
        """
        highest_similarities = {}
        for stage_analys in self.stage_analysis_objects:
            try:
                best_crystal_tuple = stage_analys.get_best_crystal(
                    self.main_stats_key
                )
            except KeyError:
                raise KeyError(
                    f"Run: {self.run_name}\n"
                    f"Key {self.main_stats_key} not found in stage {stage_analys.stage_id}."
                )
            highest_similarities[stage_analys.stage_id] = best_crystal_tuple[1]
        return highest_similarities


# ╔══════════════════════════════════════════════════════════╗
# ║                    Analysis Functions                    ║
# ╚══════════════════════════════════════════════════════════╝

def create_combined_statistics_development_plot(
        run_analysis: AnalyseRun,
        display_stage_id: bool = True,
        stage_id_y_pos: float = 1.,
        stage_id_x_offset: float = 2.5,
        statistics_name: str | None = None,
        statistics_symbol: str | None = None,
        statistics_key: str | None = None,
        x_lim: tuple[int, int] | None = None,
        y_lim: tuple[float, float] | None = None,
        save_fig: bool = False,
        legend_params: dict = dict(
            bbox_to_anchor=(0.5, 1.03), loc="lower center", fontsize=25
        ),
        y_scale: str = "linear",
        height_ratios: list[float] = [2, 1],
        fig_size: tuple[int, int] = (18, 11),
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
    if y_lim is not None:
        ax_fit.set_ylim(y_lim)
    if x_lim is not None:
        ax_fit.set_xlim(x_lim)
    ax_fit.set_yscale(y_scale)

    if statistics_name is None:
        statistics_name = run_analysis.main_stat_name

    if statistics_symbol is None:
        statistics_symbol = statistics_name[0]

    if statistics_key is None:
        statistics_key = run_analysis.main_stats_key

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
        fig.savefig(
            os.path.join(
                run_analysis.analysis_results_dir_path,
                f"{statistics_key}_development.png"
            )
        )

    return fig, ax


# ╔══════════════════════════════════════════════════════════╗
# ║                      Example Usage                       ║
# ╚══════════════════════════════════════════════════════════╝

def main(run_dir: str):
    # Load the run
    run_results = RunResults(
        run_dir=run_dir
    )

    # Create the analysis object
    analyse_run = AnalyseRun(
        run_results=run_results,
        # main_stats_key="soap_similarity_strong_RBFSimilarity",
        main_stat_name="similarity",
        analysis_results_dir_name="analysis_results",
        cell_bounds=None,
        target_crystal_id=1,
    )

    # create_combined_statistics_development_plot(
    #     analyse_run,
    #     display_stage_id=True,
    #     stage_id_x_offset=0.85,
    #     stage_id_y_pos=1.15,
    #     statistics_name="Ref. Similarity",
    #     statistics_symbol="S$_\\text{r}$",
    #     save_fig=False,
    #     y_lim=(-0.1, 1.1),
    #     legend_params=dict(
    #         bbox_to_anchor=(0.4, 1.03), loc="lower center", fontsize=25
    #     )
    # )
    # plt.show()


    fig, ax = create_combined_statistics_development_plot(
        analyse_run,
        display_stage_id=False,
        statistics_key="Volume",
        statistics_symbol="V",
        statistics_name="Log of",
        save_fig=True,
        legend_params=dict(
            bbox_to_anchor=(0.5, 1.03), loc="lower center", fontsize=25
        ),
        fig_size=(18, 9)
    )
    ax[0].set_yscale("log")
    ax[0].set_ylim(1, 3500)
    ax[0].set_ylabel("log(V)")
    ax[0].hlines(
        y=[512, 2744],
        xmin=[0, 139],
        xmax=[116, 144],
        color="black", linestyle="-.", label="Max volume", linewidth=5
    )
    ax[0].legend(
        bbox_to_anchor=(0.5, 1.03), loc="lower center", fontsize=25, ncols=4
    )
    fig.savefig(
        os.path.join(
            analyse_run.analysis_results_dir_path,
            f"Volume_development.png"
        )
    )

    # Get the analysis results dict
    analysis_results_dict = analyse_run.get_analysis_results_dict()
    import pprint
    pprint.pprint(analysis_results_dict)

if __name__ == "__main__":

    import os
    import sys

    try:
        run_dir = sys.argv[1]
    except IndexError:
        print("Please use as: python path/to/script.py path/to/run_dir")
        sys.exit(1)

    if not os.path.exists(run_dir):
        print("Path does not exist")
        sys.exit(1)

    main(run_dir)

