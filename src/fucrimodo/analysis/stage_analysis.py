from typing import Any, Callable, Literal
import pandas as pd
import warnings

from datetime import datetime

from .utils import get_end_time_from_info, get_start_time_from_info, load_dict_from_file
from ..customs.ga_stage.analysis import load_ga_stage_attributes

# ╔══════════════════════════════════════════════════════════╗
# ║                        Data Class                        ║
# ╚══════════════════════════════════════════════════════════╝


class StageData:
    """Collect and structure data produced during a single stage.

    Data is loaded from ``dir_path``. Stage-specific attributes are
    produced by the loader registered for the stage's ``type`` in
    ``stage_attribute_loader``. Custom stage types can be supported by
    adding a suitable loader to that mapping. Loaders recieve the ``dir_path``
    and the ``info_dict`` as arguments.

    Atomic structures are not loaded here. To inspect them, use ASE database
    tools on the ``structures.db`` file in the run directory, e.g.
    ``ase db <path_to_db> -w`` for an interactive web view.

    :param dir_path: Path to the directory where the stage was saved.
    :param stage_attribute_loader: Mapping from stage type name to a callable
        that accepts ``(dir_path, info_dict)`` and returns a dictionary of
        stage-specific attributes. Defaults to
        ``{"GAStage": load_ga_stage_attributes}``.

    :raises FileNotFoundError: If ``info.json`` is missing from ``dir_path``.
    :raises ValueError: If the stage type is not a key in
        ``stage_attribute_loader``.
    :raises AssertionError: If ``info.json`` does not contain a ``type`` entry.
    """

    def __init__(
        self,
        dir_path: str,
        stage_attribute_loader: dict[str, Callable[[str, dict], dict]] = {
            "GAStage": load_ga_stage_attributes
        },
    ) -> None:
        self._dir_path = dir_path

        # Load the info dict of the stage, it can then be used to get the
        # stage name, id, description, type and other data
        self._info_dict = load_dict_from_file(dir_path, "info.json")

        assert "type" in self._info_dict
        if self.type not in stage_attribute_loader:
            raise ValueError(
                f"The stage stored at {dir_path} is not one of the accepted types. Please read the docs on how to add custom stage types."
            )

        self._stage_attribute_loader = stage_attribute_loader

    @property
    def dir_path(self) -> str:
        """Path to the stage directory."""
        return self._dir_path

    @property
    def name(self) -> str:
        """Name of the stage."""
        return str(self._info_dict["name"])

    @property
    def description(self) -> str:
        """Description of the stage."""
        return str(self._info_dict["description"])

    @property
    def start_time(self) -> datetime:
        """Start time of the stage.

        :return: The stage start time as a :class:`datetime.datetime`.
        :raises KeyError: If ``start_time`` is not present in the info dict.
        """
        if not hasattr(self, "_start_time"):
            self._start_time = get_start_time_from_info(self._info_dict)
        return self._start_time

    @property
    def end_time(self) -> datetime:
        """End time of the stage.

        :return: The stage end time as a :class:`datetime.datetime`.
        :raises KeyError: If ``end_time`` is not present in the info dict.
        """
        if not hasattr(self, "_end_time"):
            self._end_time = get_end_time_from_info(self._info_dict)
        return self._end_time

    @property
    def total_runtime(self) -> str:
        """Total runtime of the run as a string, as defined in :attr:`info_dict`."""
        if not hasattr(self, "_total_runtime"):
            try:
                self._total_runtime = str(self._info_dict["total_runtime"])
            except KeyError:
                warnings.warn(
                    "No total_runtime key found in the info.json file. Returning 'Not available.'."
                )
                self._total_runtime = "Not available."
        return self._total_runtime

    @property
    def total_runtime_ms(self) -> int:
        """Total runtime of the run in milliseconds.

        Uses the ``total_runtime_ms`` entry from :attr:`info_dict` if available.

        :return: The runtime in milliseconds, or ``0`` if the key is missing.
        """
        if not hasattr(self, "_total_runtime_ms"):
            if "total_runtime_ms" in self._info_dict:
                total_runtime_ms = int(self._info_dict["total_runtime_ms"])  # type: ignore
            else:
                warnings.warn(
                    "No total_runtime_ms key found in the info.json file. Returning 0."
                )
                total_runtime_ms = 0

            self._total_runtime_ms = total_runtime_ms

        return self._total_runtime_ms

    @property
    def id(self) -> int:
        """Unique stage identifier from the info dict.

        :return: The stage id as an integer.
        :raises AssertionError: If the id is not an integer.
        """
        # Ensure that the id is an integer
        assert type(self._info_dict["id"]) == int, "The stage id is not an integer."
        return self._info_dict["id"]

    @property
    def type(self) -> str:
        return str(self._info_dict["type"])

    @property
    def n_generations(self) -> int:
        """Number of generations that the GAstage performed."""
        assert (
            type(self._info_dict["n_generations"]) == int
        ), "The key 'n_generations' was not an integer."
        return self._info_dict["n_generations"]

    @property
    def fitness_statistics(self) -> pd.DataFrame:
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
            fit_dict = load_dict_from_file(self.dir_path, "fitnesses.json")

            assert (
                type(fit_dict["results"]) == list
            ), "The results entry in the fitnesses.json file is not a list."

            # Load each of the results entries in a Dataframe
            for i in range(len(fit_dict["results"])):
                fit_dict["results"][i] = pd.DataFrame(fit_dict["results"][i])

            # Create the fitnesses Dataframe
            self._fitnesses = pd.DataFrame(fit_dict)

        return self._fitnesses

    @property
    def stage_attributes(self) -> dict:
        """Stage-specific attributes loaded by the registered stage loader.

        :return: Dictionary of attributes returned by the loader for this
            stage's type.
        :raises KeyError: If no loader is registered for this stage's type.
        """
        if not hasattr(self, "_stage_attributes"):
            self._stage_attributes = self._stage_attribute_loader[self.type](
                self._dir_path, self._info_dict
            )
        return self._stage_attributes


# ╔══════════════════════════════════════════════════════════╗
# ║                     Analysis Methods                     ║
# ╚══════════════════════════════════════════════════════════╝


def get_modification_overview(
    stage_data: StageData, modification_type: Literal["Mutation", "Crossover"]
) -> pd.DataFrame:
    """Creates an overview table of the operators with their index, names
    and representations.

    :param stage_data: Data of the stage that should be analysed.
    :param modification_type: Type of modification that should be analysed.
        Possible are "Mutation" and "Crossover".

    :return: Overview table as DataFrame.
    """
    data = stage_data.stage_attributes[f"{modification_type.lower()}s"]

    info_df = data[["names", "reprs"]].copy()

    stats = []
    for results_df in data["results"]:
        called = results_df["called"].sum()
        failed = results_df["failed"].sum()
        survivor = results_df["survivor"].sum()
        stats.append(
            {
                "total_calls": called,
                "total_fails": failed,
                "total_survivors": survivor,
                "survivor_rate": 0.0 if called == 0 else survivor / called,
                "failed_rate": 0.0 if called == 0 else failed / called,
            }
        )

    return pd.concat([info_df.reset_index(drop=True), pd.DataFrame(stats)], axis=1)


def get_stage_overview(stage_data: StageData) -> pd.Series:
    """Creates an overview table of the stage data.

    :param stage_data: Data of the stage that should be analysed.

    :return: Overview table as DataFrame.
    """
    stage_overview = pd.Series(
        {
            "Name": stage_data.name,
            "Description": stage_data.description,
            "Type": stage_data.type,
            "N_generations": stage_data.n_generations,
            "Parent Selection": stage_data.stage_attributes["parent_selection"],
            "Survivor Selection": stage_data.stage_attributes["survivor_selection"],
            "Break Condition": stage_data.stage_attributes["break_condition"],
            "Parent Ratio": stage_data._info_dict["parent_ratio"],
            "N_fit": len(stage_data.fitness_statistics),
            "N_mut": len(stage_data.stage_attributes["mutations"]),
            "N_cross": len(stage_data.stage_attributes["crossovers"]),
            "Total Runtime": stage_data.total_runtime,
        },
    )

    return stage_overview
