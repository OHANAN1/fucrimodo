import os
import ase
from fucrimodo.core.utils import ase_database_tools as db_tools
import json
from typing import Any, Callable
import numpy as np
from ase.visualize.plot import plot_atoms
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
from fucrimodo.analysis.stage_analysis import StageData

class RunData():
    """Collects and structures the data that was collected during a given run.

    :param run_dir: Path to the directory where the run was saved
        and where the results are stored. Must contain a 'info.json'
        and a 'crystals.db' file.

    :raises FileNotFoundError: One of the expected files was not found at
        path :data:`dir_path`. Expected files are 'info.json' and
        'crystals.db'.
    """
    def __init__(
        self, 
        dir_path: str,
    ) -> None:
        self._dir_path = dir_path

        # Load the info dict of the run, it can then be used to get the
        # run name, description, type and other data
        self._info_dict = self.__load_dict_from_file("info.json")

        # Load the crystals and key value pairs from the crystals database
        self._crystals, self._key_value_pairs = self.__get_crystal_data()

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
    def crystals(self) -> list[ase.Atoms]:
        """
        List of all atoms of all stages, ordered with the stage ids.
        """
        return self._crystals

    @property
    def key_value_pairs(self) -> list[dict[str, Any]]:
        """
        List of all key value pairs of all stages, ordered the same way as 
        :attr:`RunResults.crystals`.
        """
        return self._key_value_pairs

    @property
    def description(self) -> str:
        if not hasattr(self, "_description"):
            self._description = str(self._info_dict["description"])
        return self._description

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

            assert type(glob_stats_dict["results"]) == list, \
                "The results entry in the global_statistics.json file is not a list."

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
        """Dictionary with the stage ids as keys and the StageResults as values.
        """
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

    def __load_dict_from_file(
        self, file_name: str
    ) -> dict[str, list | str | int]:
        """Load a dictionary from a json file with name :data:`file_name` from
        the stage directory :attr:`RunResults.dir_path`

        :param file_name: Name of the file that should be loaded.

        :raises AssertionError: If the file does not exist in the stage 
            directory.

        :return: The loaded dictionary.
        """
        file_path = os.path.join(self.dir_path, file_name)
        assert os.path.exists(file_path), \
            f"File {file_name} does not exist in {self.dir_path}."

        with open(file_path, "r") as f:
            stage_dict = json.load(f)
        return stage_dict

    def __get_crystal_data(
            self
        ) -> tuple[list[ase.Atoms], list[dict[str, Any]]]:
        """Collects the crystals and key value pairs from the crystals database.

        The data is located at :data:`crystals_db_path`.

        :raises ValueError: If the crystals.db file does not exist in the
            directory of the run.

        :return: A tuple with the crystals and key value pairs dictionaries.
        """
        db_path = os.path.join(self.dir_path, "crystals.db")
        assert os.path.exists(db_path), \
            f"File crystals.db does not exist in {self.dir_path}."

        crystals_db = db_tools.connect_to_existing_database(db_path)
        db_data = db_tools.get_crystals_and_key_value_pairs_from_database(
            crystals_db
        )
        return db_data


def get_global_statistics_overview(run_data: RunData) -> pd.DataFrame:
    """Creates a pd.Dataframe that can be printed as a table for the
    overview of the global statistics of the run.

    :param run_data: The RunData object that contains the data of the run.

    :return: The overview table.
    """
    info_df = pd.DataFrame(
        run_data.global_statistics, 
        columns=["names", "functions"] # type: ignore
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
    y_keys: list[str] = ["min", "max", "avg"]
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
    results_df.plot(
        ax=ax,
        x="gen",
        y=y_keys,
        title=f"Global Statistic: {name}"
    )

    # Set labels of the plot
    ax.set_xlabel("Generation")
    ax.set_ylabel(y_label)


def get_best_crystal_tuple(
    run_data: RunData,
    global_statistics_index: int,
    invert: bool = False
) -> tuple[ase.Atoms, float, dict[str, Any]]:
    """Returns the crystal with the highest value for a specific global
    statistics at the provided index. To see the available statistic 
    indices and their names use :meth:`get_analysis_selection_table` with
    the analysis type "crystals".

    :param statistics_index: Index of the global statistics that should be
        used to select the best crystal.
    :param invert: If False the sorting will be inverted, meaning the 
        crystal with the lowest value of the desired statistic is returned.

    :return:

        - best crystal
        - statistics value of crystal for the given statistics key
        - key value pairs of the best crystal
    """
    stat_values = []
    selected_key = list(run_data.global_statistics["names"])[
        global_statistics_index
    ]
    assert selected_key in run_data.key_value_pairs[0].keys(), (
        f"Key {selected_key} not in key value pairs of crystal db."
    )
    stat_values = [
        key_value_pair[selected_key]
        for key_value_pair in run_data.key_value_pairs
    ]

    if invert:
        best_index = np.argmin(stat_values)
    else:
        best_index = np.argmax(stat_values)

    return (run_data.crystals[best_index],
            stat_values[best_index],
            run_data.key_value_pairs[best_index])

def get_best_crystal_overview(
    run_data: RunData,
    global_stats_row: int
    ) -> pd.DataFrame:
    """Creates a pd.Dataframe that can be printed as a table for the
    overview of the best crystal of the run.

    :param run_data: The RunData object that contains the data of the run.
    :param global_stats_row: The row index of the global statistics that
        should be used to select the best crystal.

    :return: The overview table.
    """
    best_crystal = get_best_crystal_tuple(run_data, 0)
    crystal, best_value, key_val_pair = best_crystal

    # get associated global statistics name
    global_stat_name = run_data.global_statistics.at[global_stats_row, "names"]

    overview = pd.DataFrame({
        "formula": [crystal.get_chemical_formula()],
        "best_value": best_value,
        "volume": [crystal.get_volume()],
        "n_atoms": [len(crystal)],
        "atomic_density": [crystal.get_volume() / len(crystal)],
    },
    index=[0] # type: ignore
    )

    for key, value in key_val_pair.items():
        overview[key] = value

    return overview


def visualize_best_crystal(
    run_data: RunData,
    statistics_index: int,
    ax: Axes | None = None,
    notebook_mode: bool = False
) -> None:
    from ase.visualize import view

    # Get the best crystal tuple
    best_crystal_tuple = get_best_crystal_tuple(
        run_data,
        global_statistics_index=statistics_index
    )
    crystal, best_value, key_val_pair = best_crystal_tuple

    print(f"Best crystal value: {best_value}")
    print(f"Key value pairs: ")
    from tabulate import tabulate
    table = tabulate(
        key_val_pair.items(),
        headers=["Key", "Value"],
    )
    print(table)

    # If ax is provided plot the crystal, else view it
    if ax is not None:
        plot_atoms(crystal, ax=ax)
        ax.set_title(f"Best Crystal: {best_value}")
    else:
        # Embed the crystal in notebook or in a new window
        if notebook_mode:
            view(crystal, viewer="x3d")
        else:
            view(crystal)


def get_run_overview(run_data: RunData) -> pd.DataFrame:
    """Creates a pd.Dataframe that can be printed as a table for the 
    overview of the run with an index and further information to enable 
    the selection of the desired stage.

    :return: The overview table.
    """
    overview = pd.DataFrame({
        "name": run_data.name,
        "description": run_data.description,
        "n_stages": run_data.n_stages,
        "total_generations": run_data.total_generations
    },
    index=[0] # type: ignore
    )
    return overview


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

    run_data = RunData(run_dir)

    print("________________________________________________________")
    print("Run Overview:")
    print(get_run_overview(run_data))
    print()
    print("________________________________________________________")
    print("Global Statistics Overview:")
    print(get_global_statistics_overview(run_data))
    print()
    print("________________________________________________________")
    print("Best Crystal Overview:")
    print(get_best_crystal_overview(run_data, 0))

    plot_global_statistics(run_data, 0)
    plot_global_statistics(run_data, 1)

    visualize_best_crystal(run_data, 0)
    plt.show()

