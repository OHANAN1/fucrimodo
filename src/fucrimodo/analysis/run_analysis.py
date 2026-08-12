import os
import ase
from fucrimodo.utils import ase_tools
import json
from typing import Any
import numpy as np
from ase.visualize.plot import plot_atoms
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
from fucrimodo.analysis.stage_analysis import StageData


class RunData:
    """Collects and structures the data that was collected during a given run.

    :param run_dir: Path to the directory where the run was saved
        and where the results are stored. Must contain a 'info.json'
        and a 'structures.db' file.

    :raises FileNotFoundError: One of the expected files was not found at
        path :data:`dir_path`. Expected files are 'info.json' and
        'structures.db'.
    """

    def __init__(
        self,
        dir_path: str,
    ) -> None:
        self._dir_path = dir_path

        # Load the info dict of the run, it can then be used to get the
        # run name, description, type and other data
        self._info_dict = self.__load_dict_from_file("info.json")

        # Load the structures and key value pairs from the structures database
        self._structures, self._key_value_pairs = self.__get_structures_data()

    @property
    def dir_path(self) -> str:
        """Path to the run directory."""
        return self._dir_path

    @property
    def name(self) -> str:
        if not hasattr(self, "_name"):
            self._name = str(self._info_dict["name"])
        return self._name

    @property
    def description(self) -> str:
        if not hasattr(self, "_description"):
            self._description = str(self._info_dict["description"])
        return self._description

    @property
    def start_time(self) -> str:
        if not hasattr(self, "_start_time"):
            try:
                self._start_time = str(self._info_dict["start_time"])
            except KeyError:
                self._start_time = "Not available."
        return self._start_time

    @property
    def start_time_ms(self) -> str:
        if not hasattr(self, "_start_time_ms"):
            try:
                self._start_time = str(self._info_dict["start_time_ms"])
            except KeyError:
                self._start_time = "Not available."
        return self._start_time

    @property
    def end_time(self) -> str:
        if not hasattr(self, "_end_time"):
            try:
                self._end_time = str(self._info_dict["end_time"])
            except KeyError:
                self._end_time = "Not available."

        return self._end_time

    @property
    def total_runtime(self) -> str:
        if not hasattr(self, "_total_runtime"):
            try:
                self._total_runtime = str(self._info_dict["total_runtime"])
            except KeyError:
                self._total_runtime = "Not available."

        return self._total_runtime

    @property
    def structures(self) -> list[ase.Atoms]:
        """
        List of all structures of all stages, ordered with the stage ids.
        """
        return self._structures

    @property
    def key_value_pairs(self) -> list[dict[str, Any]]:
        """
        List of all key value pairs of all stages, ordered the same way as
        :attr:`RunResults.structures`.
        """
        return self._key_value_pairs

    @property
    def global_statistics(self) -> pd.DataFrame:
        """The global statistics info and statistics that where tracked.

        A pandas Dataframe with keys `names`, `function_names` and
        `results`.
        Each row corresponds to a specific global statistic.
        The results entry is a Dataframe with columns `min`, `max`,
        `avg` and `std` of the statistic and the `gen` and `stage_id`
        where it was tracked.
        """
        if not hasattr(self, "_global_statistics_log"):
            # Load the global statistics dict from global_statistics.json
            glob_stats_dict = self.__load_dict_from_file("global_statistics.json")

            assert (
                type(glob_stats_dict["results"]) == list
            ), "The results entry in the global_statistics.json file is not a list."

            # Load each of the results entries in a seperate Dataframe
            for i in range(len(glob_stats_dict["results"])):
                glob_stats_dict["results"][i] = pd.DataFrame(
                    glob_stats_dict["results"][i]
                )

            # Create the Dataframe
            self._global_statistics = pd.DataFrame(glob_stats_dict)
        return self._global_statistics

    @property
    def total_generations(self) -> int:
        """Returns the total number of generation of the whole run."""
        # Get the first global stat entry and get the max gen
        # This is the total number of generations, since the global
        # statistics track the global gen number
        total_gen = np.max(self.global_statistics.loc[0, "results"]["gen"])
        return total_gen

    @property
    def stages(self) -> dict[int, StageData]:
        """Dictionary with the stage ids as keys and the StageResults as values."""
        if not hasattr(self, "_stages"):
            # Checks if stages where performed
            stage_hist = self._info_dict["stage_history"]
            assert type(stage_hist) == dict, "The stage history is not a dictionary."
            assert len(stage_hist["ID"]) > 0, "The stage history is empty."

            # Get the stages from the specified directories and load them
            self._stages = {}
            for i in range(len(stage_hist["ID"])):
                stage_id = stage_hist["ID"][i]

                # Add the relative path to the stage directory
                stage_path = os.path.join(
                    self.dir_path, stage_hist["relative_save_path"][i]
                )

                # Load the stage data
                self._stages[stage_id] = StageData(stage_path)

        return self._stages

    @property
    def n_stages(self) -> int:
        """Number of stages that where performed."""
        return len(self.stages)

    def __load_dict_from_file(self, file_name: str) -> dict[str, list | str | int]:
        """Load a dictionary from a json file with name :data:`file_name` from
        the stage directory :attr:`RunResults.dir_path`

        :param file_name: Name of the file that should be loaded.

        :raises AssertionError: If the file does not exist in the stage
            directory.

        :return: The loaded dictionary.
        """
        file_path = os.path.join(self.dir_path, file_name)
        assert os.path.exists(
            file_path
        ), f"File {file_name} does not exist in {self.dir_path}."

        with open(file_path, "r") as f:
            stage_dict = json.load(f)
        return stage_dict

    def __get_structures_data(self) -> tuple[list[ase.Atoms], list[dict[str, Any]]]:
        """Collects the structures and key value pairs from the structures database.

        The data is located at :data:`structures_db_path`.

        :raises ValueError: If the structures.db file does not exist in the
            directory of the run.

        :return: A tuple with the structures and key value pairs dictionaries.
        """
        db_path = os.path.join(self.dir_path, "structures.db")
        assert os.path.exists(
            db_path
        ), f"File structures.db does not exist in {self.dir_path}."

        structures_db = ase_tools.connect_to_existing_database(db_path)
        db_data = ase_tools.get_structures_and_key_value_pairs_from_database(
            structures_db
        )
        return db_data


def get_global_statistics_overview(run_data: RunData) -> pd.DataFrame:
    """Creates a pd.Dataframe that can be printed as a table for the
    overview of the global statistics of the run.

    :param run_data: The RunData object that contains the data of the run.

    :return: The overview table.
    """
    info_df = pd.DataFrame(
        run_data.global_statistics, columns=["names", "functions"]  # type: ignore
    )
    result_dfs = run_data.global_statistics["results"]

    # Add the absolute maximum and minimum of the statistics to the overview
    for i, result_df in enumerate(result_dfs):
        info_df.at[i, "max"] = result_df["max"].max()
        info_df.at[i, "min"] = result_df["min"].min()

    return info_df


def plot_global_statistics(
    run_data: RunData,
    row: int,
    ax: Axes | None = None,
    y_label: str = "Value",
    y_keys: list[str] = ["min", "max", "avg"],
) -> None:
    """Performs the desired analysis.

    :param run_data: The RunData object that contains the data of the run.
    :param row: Row index of the statistic that should be plotted from the
        :attr:`RunData.global_statistics` dataframe.
    :param ax: Optional axis to plot the results on. If None a new figure
        will be created.

    :returns: The result of the analysis.
    """
    # Get the name and results dataframe of the selected global statistic
    name = run_data.global_statistics.at[row, "names"]
    results_df = run_data.global_statistics.loc[row, "results"]

    # Define a plot if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # Plot the selected results data
    results_df.plot(ax=ax, x="gen", y=y_keys, title=f"Global Statistic: {name}")

    # Set labels of the plot
    ax.set_xlabel("Generation")
    ax.set_ylabel(y_label)

    # Plot vertical lines at the stage boundaries
    # Get the initial stage id
    current_stage_id = results_df["stage_id"].iloc[0]
    for i in range(len(results_df["stage_id"])):
        stage_id = results_df["stage_id"].iloc[i]
        # Check if the selected stage id is different from the current one
        if stage_id != current_stage_id:
            # Plot a vertical line at the current generation
            ax.axvline(
                x=results_df["gen"].iloc[i],
                color="black",
                linestyle="--",
            )

            # Update the current stage id
            current_stage_id = stage_id

    # Add the legend
    ax.legend()


def get_best_structure_tuple(
    run_data: RunData, global_statistics_row: int, invert: bool = False
) -> tuple[ase.Atoms, float, dict[str, Any]]:
    """Returns the structures with the highest value for a specific global
    statistics at the provided index. To see the available statistic
    indices and their names use :meth:`get_analysis_selection_table` with
    the analysis type "structures".

    :param statistics_index: Index of the global statistics that should be
        used to select the best structures.
    :param invert: If False the sorting will be inverted, meaning the
        structures with the lowest value of the desired statistic is returned.

    :return:

        - best structure
        - statistics value of structure for the given statistics key
        - key value pairs of the best structure
    """
    stat_values = []
    selected_key = list(run_data.global_statistics["names"])[global_statistics_row]
    assert (
        selected_key in run_data.key_value_pairs[0].keys()
    ), f"Key {selected_key} not in key value pairs of structures db."
    stat_values = [
        key_value_pair[selected_key] for key_value_pair in run_data.key_value_pairs
    ]

    if invert:
        best_index = np.argmin(stat_values)
    else:
        best_index = np.argmax(stat_values)

    return (
        run_data.structures[best_index],
        stat_values[best_index],
        run_data.key_value_pairs[best_index],
    )


def get_best_structure_overview(
    run_data: RunData, global_stats_row: int
) -> pd.DataFrame:
    """Creates a pd.Dataframe that can be printed as a table for the
    overview of the best structures of the run.

    :param run_data: The RunData object that contains the data of the run.
    :param global_stats_row: The row index of the global statistics that
        should be used to select the best structures.

    :return: The overview table.
    """
    best_structure = get_best_structure_tuple(run_data, global_stats_row)
    structure, best_value, key_val_pair = best_structure

    overview = pd.DataFrame(
        {
            "formula": [structure.get_chemical_formula()],
            "best_value": best_value,
            "volume": [structure.get_volume()],
            "n_atoms": [len(structure)],
            "atomic_density": [structure.get_volume() / len(structure)],
        },
        index=[0],  # type: ignore
    )

    for key, value in key_val_pair.items():
        overview[key] = value

    return overview


def visualize_best_structure(
    run_data: RunData,
    global_statistics_row: int,
    ax: Axes | None = None,
    notebook_mode: bool = False,
) -> None:
    from ase.visualize import view

    # Get the best structure tuple
    best_structure_tuple = get_best_structure_tuple(
        run_data=run_data, global_statistics_row=global_statistics_row
    )
    structure, best_value, key_val_pair = best_structure_tuple

    # If ax is provided plot the structure, else view it
    if ax is not None:
        plot_atoms(structure, ax=ax)
        ax.set_title(f"Best structure: {best_value}")
    else:
        # Embed the structure in notebook or in a new window
        if notebook_mode:
            print("Plotting in notebook mode.")
            view(structure, viewer="x3d")
        else:
            view(structure)


def get_run_overview(run_data: RunData) -> pd.DataFrame:
    """Creates a pd.Dataframe that can be printed as a table for the
    overview of the run with an index and further information to enable
    the selection of the desired stage.

    :return: The overview table.
    """
    overview = pd.DataFrame(
        {
            "name": run_data.name,
            "description": run_data.description,
            "n_stages": run_data.n_stages,
            "total_generations": run_data.total_generations,
            "total_runtime": run_data.total_runtime,
        },
        index=[0],  # type: ignore
    )
    return overview
