import os
import pandas as pd
import json

from fucrimodo.analysis.run_analysis import (
    RunData,
    get_global_statistics_overview
)


class MultiRunData():
    def __init__(self, multi_run_dir: str):
        self.multi_run_dir = os.path.abspath(multi_run_dir)

        # Create a list of RunData objects
        self.runs = self.__load_runs_from_dir()

        # Load the info.json of the multi run, if it exists
        self._info_dict = None
        info_path = os.path.join(self.multi_run_dir, "info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                self.info_dict = json.load(f)

    @property
    def total_runtime(self) -> str:
        """Returns the total runtime of all loaded runs."""
        if self.info_dict is not None:
            return self.info_dict["total_runtime"]
        else:
            return "N/A"
        

    @property
    def name_list(self) -> list[str]:
        """Returns the names of all loaded runs."""
        return [run.name for run in self.runs]

    @property
    def description_list(self) -> list[str]:
        """Returns the descriptions of all loaded runs."""
        return [run.description for run in self.runs]

    @property
    def n_runs(self) -> int:
        """Returns the number of loaded runs."""
        return len(self.runs)

    @property
    def n_stages_list(self) -> list[int]:
        """Returns the number of stages for each loaded run."""
        return [run.n_stages for run in self.runs]

    @property
    def total_runtime_list(self) -> list[str]:
        """Returns the total runtime for each loaded run."""
        return [run.total_runtime for run in self.runs]

    @property
    def total_generations_list(self) -> list[int]:
        """Returns the total number of generations for each loaded run."""
        return [run.total_generations for run in self.runs]

    def __load_runs_from_dir(self) -> list[RunData]:
        """Tries to load all directories in the multi_run_dir as RunData objects.

        If a directory is not a valid run, it will be skipped and an 
        error message will be printed.
        """
        runs = []
        for run_dir in os.listdir(self.multi_run_dir):
            try:
                run = RunData(os.path.join(self.multi_run_dir, run_dir))

            except Exception as e:
                print(f"Error loading run {run_dir}: {e}")
                continue

            runs.append(run)

        return runs


def get_multi_run_overview(multi_run_data: MultiRunData) -> pd.DataFrame:
    """Creates a pd.Dataframe that can be printed as a table for the 
    overview of all runs.

    :return: The overview table.
    """
    overview = pd.DataFrame({
        "name": multi_run_data.name_list,
        "description": multi_run_data.description_list,
        "n_stages": multi_run_data.n_stages_list,
        "total_generations": multi_run_data.total_generations_list,
        "total_runtime": multi_run_data.total_runtime_list
    })
    return overview


def get_all_global_statistics_overview(
    multi_run_data: MultiRunData,
    round_decimals: int = 5,
) -> pd.DataFrame:
    """Creates a pd.Dataframe that can be printed as a table for the
    overview of the global statistics of all the runs.

    :param run_data: The RunData object that contains the data of the run.

    :return: The overview table.
    """
    # Get the global stats overview of all runs
    global_stats_list = []
    for run_index, run_data in enumerate(multi_run_data.runs):
        global_stats = get_global_statistics_overview(run_data)

        # Reshape the global stats overview to have the stats names as columns
        # and the min, max value as list of rows
        reshaped_global_stats = {}
        for i, global_stats_name in enumerate(global_stats["names"]):
            reshaped_global_stats["run"] = run_data.name

            # Convert the min, max values to a string so they can be displayed
            # in the same cell
            reshaped_global_stats[f"{global_stats_name}_min_max"] = (
                f"{global_stats["min"][i]:.{round_decimals}f}, "
                f"{global_stats["max"][i]:.{round_decimals}f}"
            )

        # Create a pd.DataFrame from the reshaped global stats
        # This is done so runs with different stats can be concatenated later
        # in a way that non existing stats are filled with NaN
        global_stats_df = pd.DataFrame(
            reshaped_global_stats,
            index=[run_index]  # type: ignore
        )
        global_stats_list.append(global_stats_df)

    # Concatenate all the global stats DataFrames
    global_stats_overview = pd.concat(global_stats_list)

    return global_stats_overview
