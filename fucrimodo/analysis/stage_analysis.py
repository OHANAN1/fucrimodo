import os
import json
import pandas as pd
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from functools import partial


# ╔══════════════════════════════════════════════════════════╗
# ║                        Data Class                        ║
# ╚══════════════════════════════════════════════════════════╝

class StageData():
    """Collects and structures the data that was collected during a given
    stage.

    :param stage_dir: Path to the directory where the stage was saved

    :raises FileNotFoundError: One of the expected files was not found at 
        path :data:`dir_path`. Expected files are 'info.json', 
        'fitnesses.json', 'mutations.json', 'crossovers.json'.
    :raises ValueError: If the expected files do not contain the expected
        data.
    """
    def __init__(
        self, 
        dir_path: str,
    ) -> None:
        self._dir_path = dir_path

        # Load the info dict of the stage, it can then be used to get the
        # stage name, id, description, type and other data
        self._info_dict = self.__load_dict_from_file("info.json")

    @property
    def dir_path(self) -> str:
        """Path to the stage directory."""
        return self._dir_path

    @property
    def name(self) -> str:
        return str(self._info_dict["name"])

    @property
    def description(self) -> str:
        return str(self._info_dict["description"])

    @property
    def id(self) -> int:
        # Ensure that the id is an integer
        assert type(self._info_dict["id"]) == int, \
            "The stage id is not an integer."
        return self._info_dict["id"]

    @property
    def parent_selection(self) -> str:
        return str(self._info_dict["parent_selection"])

    @property
    def survivor_selection(self) -> str:
        return str(self._info_dict["survivor_selection"])

    @property
    def break_condition(self) -> str:
        return str(self._info_dict["break_condition"])

    @property
    def type(self) -> str:
        return str(self._info_dict["type"])

    @property
    def n_generations(self) -> int:
        """Number of generations that the GAstage performed."""
        assert type(self._info_dict["n_generations"]) == int, \
            "The key 'n_generations' was not an integer." 
        return self._info_dict["n_generations"]

    @property
    def fitnesses(self) -> pd.DataFrame:
        """The fitness information and statistics that where tracked during the stage.

        A Dataframe with columns `names`, `weights`, `reprs`, `hashes` and
        `results`.
        Each row corresponds to a specific fitness operator.
        The results entries are dataframes with columns `max`, `min`,
        `avg`, `std` and `gen`.
        """
        # Check if the fitnesses where already loaded
        if not hasattr(self, "_fitnesses"):
            # Load the fitnesses dict from the fitnesses.json file
            fit_dict = self.__load_dict_from_file("fitnesses.json")

            assert type(fit_dict["results"]) == list, \
                "The results entry in the fitnesses.json file is not a list."

            # Load each of the results entries in a Dataframe
            for i in range(len(fit_dict["results"])):
                fit_dict["results"][i] = pd.DataFrame(fit_dict["results"][i])

            # Create the fitnesses Dataframe
            self._fitnesses = pd.DataFrame(fit_dict)

        return self._fitnesses

    @property
    def mutations(self) -> pd.DataFrame:
        """The mutation information and statistics that where tracked during the stage.

        A pandas dataframe with keys `names`, `weights`, `reprs`, `hashes` and
        `results`.
        Each row corresponds to a specific mutation operator.
        The results entry is a Dataframe with columns `called`, `failed`,
        `survivor` and `gen`.
        """
        # Check if the mutations where already loaded
        if not hasattr(self, "_mutations"):
            # Get the dict from the mutations.json file
            mut_dict = self.__load_dict_from_file("mutations.json")

            assert type(mut_dict["results"]) == list, \
                "The results entry in the mutations.json file is not a list."

            # Load each of the results entries in a Dataframe
            for i in range(len(mut_dict["results"])):
                mut_dict["results"][i] = pd.DataFrame(mut_dict["results"][i])

            # Create the mutations Dataframe
            self._mutations = pd.DataFrame(mut_dict)
        return self._mutations

    @property
    def crossovers(self) -> pd.DataFrame:
        """The crossover information and statistics that where tracked during the stage.

        A pandas Dataframe with keys `names`, `weights`, `reprs`, `hashes` and 
        `results`.
        Each row corresponds to a specific crossover operator.
        The results entry is a Dataframe with columns `called`, `failed`,
        `survivor` and `gen`.
        """
        # Check if the crossovers where already loaded
        if not hasattr(self, "_crossovers"):
            # Load the crossovers dict from the crossovers.json file
            cross_dict = self.__load_dict_from_file("crossovers.json")

            assert type(cross_dict["results"]) == list, \
                "The results entry in the crossovers.json file is not a list."

            # Load each of the results entries in a Dataframe
            for i in range(len(cross_dict["results"])):
                cross_dict["results"][i] = pd.DataFrame(cross_dict["results"][i])

            # Create the crossovers Dataframe
            self._crossovers = pd.DataFrame(cross_dict)

        return self._crossovers

    def __load_dict_from_file(
        self, file_name: str
    ) -> dict[str, list | str | int]:
        """Load a dictionary from a json file with name :data:`file_name` from
        the stage directory :attr:`StageResults.dir_path`

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

# ╔══════════════════════════════════════════════════════════╗
# ║                     Analysis Methods                     ║
# ╚══════════════════════════════════════════════════════════╝


def get_fitness_overview(stage_data: StageData) -> pd.DataFrame:
    """Creates an overview table of the fitness operators with their index,
    names and representations.

    :param stage_data: Data of the stage that should be analysed.

    :return: Overview table as string.
    """
    info_df = pd.DataFrame(
        stage_data.fitnesses,
        columns=["names", "reprs"] # type: ignore
    )
    results_dfs = stage_data.fitnesses["results"]

    # For each fitness operator, get the absolute maximum and minimum fitness 
    # for all generations
    for i in range(len(info_df)):
        results_df = results_dfs[i]
        info_df.loc[i, "absolute_max"] = np.max(results_df["max"])
        info_df.loc[i, "absolute_min"] = np.min(results_df["min"])

    return info_df


def plot_fitness_statistics(
    stage_data: StageData,
    row: int,
    ax: Axes | None = None,
    x_key: str = "gen",
    x_label: str = "Generation",
    y_keys: list[str] = ["max", "min", "avg"],
    y_label: str = "Fitness Value"
) -> None:
    """Analyse the fitness data of a specific fitness operator.

    :param stage_data: Data of the stage that should be analysed.
    :param ax: Matplotlib axis object to plot on.
    :param row: Index of the row in the fitness results dataframe of
        which the results data should be plotted. Use::

            print(show_fitness_overview(stage_data))

        to show the names and representations of the fitness operators.
    :param x_key: Key of the x-axis data. Normally the generation is used
        as x-axis, with the key "gen".
    :param y_keys: List of keys of the y-axis data that should be plotted.
    :param y_label: Label of the y-axis.
    """
    # Get the name and results entry of the selected fitness operator
    name = stage_data.fitnesses.at[row, "names"]
    results_df = stage_data.fitnesses.loc[row, "results"]

    # Create a new figure to plot on if no axis is given
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the selected results data
    results_df.plot(
        ax=ax,
        x=x_key,
        y=y_keys,
        title=f"Fitnesses: {name}"
    )

    # Set labels of the plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def get_modification_overview(
    stage_data: StageData, modification_type: str
) -> pd.DataFrame:
    """Creates an overview table of the operators with their index, names
    and representations.

    :param stage_data: Data of the stage that should be analysed.
    :param modification_type: Type of modification that should be analysed.
        Possible are "Mutation" and "Crossover".

    :return: Overview table as DataFrame.
    """
    if modification_type == "Mutation":
        info_df = pd.DataFrame(
            stage_data.mutations,
            columns=["names", "reprs"] # type: ignore
        )
        results_dfs = stage_data.mutations["results"]
    elif modification_type == "Crossover":
        info_df = pd.DataFrame(
            stage_data.crossovers,
            columns=["names", "reprs"] # type: ignore
        )
        results_dfs = stage_data.crossovers["results"]
    else:
        raise ValueError(
            "The given modification type is not valid for this Stage.\n"
        )

    # Calculate the total number of calls, failed calls and survivors for 
    # each operator
    for i in range(len(info_df)):
        results_df = results_dfs[i]

        # Sum up the total statistics
        total_called = results_df["called"].sum()
        total_failed = results_df["failed"].sum()
        total_survivor = results_df["survivor"].sum()

        # Calculate the rates. If no calls where made, the rates are 0
        if total_called == 0:
            survivor_rate = 0
            failed_rate = 0
        else:
            survivor_rate = total_survivor / total_called
            failed_rate = total_failed / total_called

        # Add the statistics to the info_df
        info_df.loc[i, "total_calls"] = total_called
        info_df.loc[i, "total_failed"] = total_failed
        info_df.loc[i, "total_survivors"] = total_survivor
        info_df.loc[i, "survivor_rate"] = survivor_rate
        info_df.loc[i, "failed_rate"] = failed_rate

    return info_df


def plot_modification_statistics(
    stage_data: StageData,
    modification_type: str,
    row: int,
    ax: Axes | None = None, 
    x_key: str = "gen",
    x_label: str = "Generation",
    y_keys: list[str] | None = None,
    y_label: str = "Number of Calls"
) -> None:
    """Performs the analysis for the desired analysis type.

    :param stage_data: Data class of the stage that should be analysed.
    :param modification_type: Type of modification that should be analysed.
        Possible are "Mutation" and "Crossover".
    :param row: Index of the row in the mutation or crossover results 
        dataframe of which the results data should be plotted. Use::

            print(get_modification_overview(stage_data, modification_type))

        to show the names and representations of the fitness operators.
    :param ax: Matplotlib axis object to plot on. If None, a new figure
        is created.
    :param x_key: Key of the x-axis data. Normally the generation is used
        as x-axis, with the key "gen".
    :param y_keys: List of keys of the y-axis data that should be plotted.
        If None, all columns of the results dataframe are plotted.
        To get the column names use
        :code:`self.stage_results.crossovers.loc[row, "results"].columns`.

    :raises IndexError: If the given crossover index is out of range.
    """
    # Load name and results_df depending on the analysis type
    if modification_type == "Mutation":
        name = stage_data.mutations.at[row, "names"]
        results_df: pd.DataFrame = stage_data.mutations.loc[
            row, "results"
        ]

    elif modification_type == "Crossover":
        name = stage_data.crossovers.at[row, "names"]
        results_df: pd.DataFrame = stage_data.crossovers.loc[
            row, "results"
        ]

    else:
        raise ValueError(
            "The given modification type is not valid for this Stage.\n"
        )

    # Create a new figure to plot on if no axis is given
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the selected results data
    results_df.plot(
        ax=ax,
        x=x_key,
        y=y_keys,
        title=f"{modification_type}: {name}"
    )

    # Set labels of the plot
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def get_stage_overview(stage_data: StageData) -> pd.DataFrame:
    """Creates an overview table of the stage data.

    :param stage_data: Data of the stage that should be analysed.

    :return: Overview table as DataFrame.
    """
    stage_overview = pd.DataFrame(
        {
            "Name": stage_data.name,
            "Description": stage_data.description,
            "Type": stage_data.type,
            "N_generations": stage_data.n_generations,
            "Parent Selection": stage_data.parent_selection,
            "Survivor Selection": stage_data.survivor_selection,
            "Break Condition": stage_data.break_condition,
            "Parent Ratio": stage_data._info_dict["parent_ratio"],
            "N_fit": len(stage_data.fitnesses),
            "N_mut": len(stage_data.mutations),
            "N_cross": len(stage_data.crossovers),
        },
        index=[0] # type: ignore
    )

    return stage_overview


def cli_runner(
    stage_dir: str,
    row: int | None = None,
    show: bool = False,
    verbose: bool = False,
    analysis_type: str | None = None,
    save_dir: str | None = None
):
    # Load the stage data
    stage_data = StageData(stage_dir)

    print(f"{analysis_type} Overview:")


    # Depending on the analysis type, set a different overview dataframe and
    # plot a different plot function
    if analysis_type is not None:
        # Set the row to 0 if it is None
        if row is None:
            print("Row: 0 (default)")
            row = 0
        else:
            print(f"Row: {row}")
        print()

        if analysis_type == "Fitness":
            # Get the fitness overview dataframe
            print("Fitness Overview:")
            print()
            print(get_fitness_overview(stage_data))
            plot_fitness_statistics(
                stage_data = stage_data,
                row = row
            )
            print()

        elif analysis_type == "Mutation" or analysis_type == "Crossover":
            print(f"{analysis_type} Overview:")
            print()
            print(get_modification_overview(stage_data, analysis_type))
            print()
            plot_modification_statistics(
                stage_data = stage_data,
                row = row,
                modification_type = analysis_type
            )

        else:
            raise ValueError(
                "The given analysis type is not valid for this stage. "
                "Possible are 'Fitness', 'Mutation' and 'Crossover'. "
                    "(Upper case is required)"
            )

        # Show the plot or save them to a file
        if show:
            plt.show()
        else:
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(f"{save_dir}/{analysis_type}_{row}_overview.png")

            else:
                plt.savefig(f"{analysis_type}_{row}_overview.png")
                plt.close()


    else:
        print("No analysis type given. Displaying general overview.")
        print("Stage Overview:")
        print(get_stage_overview(stage_data).T)
        print()

        print("-------------------")
        print("Fitness Overview:")
        print(get_fitness_overview(stage_data))
        print()

        print("-------------------")
        print("Mutation Overview:")
        print(get_modification_overview(stage_data, "Mutation"))
        print()

        print("-------------------")
        print("Crossover Overview:")
        print(get_modification_overview(stage_data, "Crossover"))
        print()


if __name__ == "__main__":

    import sys

    try:
        stage_dir = sys.argv[1]
    except IndexError:
        print("Please use as: python path/to/script.py path/to/stage_dir")
        sys.exit(1)

    if not os.path.exists(stage_dir):
        print("Path does not exist")
        sys.exit(1)

    cli_runner(stage_dir)

