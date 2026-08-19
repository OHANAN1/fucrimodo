import os
import warnings
from datetime import datetime
from typing import Any, Callable

import ase
import numpy as np
import pandas as pd

from ..core import Individual
from ..customs.ga_stage.analysis import load_ga_stage_attributes
from ..utils import ase_tools
from .stage_analysis import StageData
from .utils import get_end_time_from_info, get_start_time_from_info, load_dict_from_file


class RunData:
    """Collects and structures the data produced by a single run.

    The run directory is expected to contain an ``info.json`` file that must
    have an entry called ``stage_history``. By design all properties are then loaded on
    demand, this means that if they are missing in the database this
    only leads to errors when they are accessed.  Stage data is loaded from the
    subdirectories recorded in the stage history of ``info.json``.

    :param dir_path: Path to the directory where the run was saved.
    :param accepted_stage_types: List of stage type names that are accepted when
        loading stage data. Add new custom stages here, after carefully reading
        the documentation on how to implement new stages.  Defaults to
        ``["GAStage"]``.

    :raises FileNotFoundError: If ``info.json`` is not found in the run
        directory.

    :raises AssertionError: If the info dict does not have an entry for
        ``stage_history``.  Further it must be a dictionary with none-zero
        entries or entries of different lenghts.

    :raises AssertionError: If the info dict does not have an entry for
        ``stage_history``. And if the value of ``stage_history`` is a zero-lenght dictionary
        or has entries of different lenghts.
    """

    def __init__(
        self,
        dir_path: str,
        stage_attribute_loader: dict[str, Callable[[str, dict], dict]] = {
            "GAStage": load_ga_stage_attributes
        },
    ) -> None:
        self._dir_path = dir_path
        self._stage_attribute_loader = stage_attribute_loader

        # Test if its a run dir
        is_run_dir, reason = RunData.is_run_dir(dir_path)
        if not is_run_dir:
            raise ValueError(reason)

        # Only load info, the rest is hot loaded when needed
        self._info_dict = load_dict_from_file(self.dir_path, "info.json")

        # Assign stage history and ensure that its in the right format
        self._stage_history = self._info_dict["stage_history"]
        assert (
            type(self._stage_history) == dict
        ), "The stage history is not a dictionary."
        assert len(self._stage_history["ID"]) > 0, "The stage history is empty."
        assert len(self._stage_history["relative_save_path"]) == len(
            self._stage_history["ID"]
        ), "The stage history is invalide."

    @property
    def info_dict(self) -> dict:
        """Dictionary from the ``info.json`` in the run dir."""
        return self._info_dict

    @property
    def dir_path(self) -> str:
        """Path to the run directory."""
        return self._dir_path

    @property
    def name(self) -> str:
        """Name of the run as defined in :attr:`info_dict`"""
        if not hasattr(self, "_name"):
            self._name = str(self._info_dict["name"])
        return self._name

    @property
    def description(self) -> str:
        """Description of the run as defined in :attr:`info_dict`"""
        if not hasattr(self, "_description"):
            self._description = str(self._info_dict["description"])
        return self._description

    @property
    def start_time(self) -> datetime:
        """Start time of the stage.

        :raises KeyError: If ``start_time`` is not present in the info dict.
        """
        if not hasattr(self, "_start_time"):
            self._start_time = get_start_time_from_info(self._info_dict)
        return self._start_time

    @property
    def end_time(self) -> datetime:
        """End time of the stage.

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

    def __get_structures_data(self) -> tuple[list[ase.Atoms], list[dict[str, Any]]]:
        """Collects the structures and key value pairs from the structures database.

        The data is located at :data:`structures_db_path`.

        :raises ValueError: If the structures.db file does not exist in the
            directory of the run.

        :return: A tuple with the structures and key value pairs dictionaries.
        """
        db_path = os.path.join(self.dir_path, "structures.db")
        structures_db = ase_tools.connect_to_existing_database(db_path)
        db_data = ase_tools.get_structures_and_key_value_pairs_from_database(
            structures_db
        )
        return db_data

    @property
    def structures(self) -> list[ase.Atoms]:
        """The best structures of the original run.

        Structures are loaded from ``structures.db`` in the :attr:`run_dir`
        the first time this property is accessed. The list contains all
        structures from all stages, ordered by stage id.

        Also automatically loads the :attr:`key_value_pairs`

        :return: List of :class:`ase.Atoms` objects. The index of each structure
            matches the corresponding entry in :attr:`key_value_pairs`.

        :raises FileNotFoundError: If ``structures.db`` does not exist in the
            run directory.
        """
        if not hasattr(self, "_structures"):
            # Load the structures and key value pairs from the structures database
            self._structures, self._key_value_pairs = self.__get_structures_data()

        return self._structures

    @property
    def key_value_pairs(self) -> list[dict[str, Any]]:
        """Key-value pairs associated with all structures in the run.

        Key-value pairs are loaded from ``structures.db`` together with
        :attr:`structures`. Each dictionary contains the metadata and calculated
        properties of the structure at the same index in :attr:`structures`.

        :return: List of dictionaries. ``key_value_pairs[i]`` belongs to
            ``structures[i]``.
        :raises FileNotFoundError: If ``structures.db`` does not exist in the
            run directory.
        """
        if not hasattr(self, "_key_value_pairs"):
            # Load the structures and key value pairs from the structures database
            self._structures, self._key_value_pairs = self.__get_structures_data()

        return self._key_value_pairs

    @property
    def global_statistics(self) -> pd.DataFrame:
        """Global statistics tracked during the run.

        A :class:`pandas.DataFrame` with columns ``names``, ``function_names``
        and ``results``. Each row corresponds to a specific global statistic.
        The ``results`` entry is a :class:`pandas.DataFrame` with columns
        ``min``, ``max``, ``avg``, ``std``, ``gen`` and ``stage_id``.

        Access results for each statistic with:

        .. code-block:: python

            entry = 0
            statistics_name = global_statistics.loc[entry, "names"]
            statistics = global_statistics.loc[entry, "results"]

        """
        if not hasattr(self, "_global_statistics_log"):
            # Load the global statistics dict from global_statistics.json
            glob_stats_dict = load_dict_from_file(
                self.dir_path, "global_statistics.json"
            )

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
    def n_generations(self) -> int:
        """Total number of generations of the whole run.

        Calculated from the maximum value of the ``gen`` column in the
        ``results`` of the first global statistic.

        :return: The highest generation number.

        :raises KeyError: If the first global statistic has no ``gen`` column.
        """
        # Get the first global stat entry and get the max gen
        # This is the total number of generations, since the global
        # statistics track the global gen number
        return self.global_statistics.loc[0, "results"]["gen"].max()

    @property
    def stages(self) -> list[StageData]:
        """All stages performed in the run, sorted by stage id.

        Stage directories are read from ``stage_history["relative_save_path"]``
        in :attr:`info_dict`. Each directory is loaded as a :class:`StageData`
        instance and sorted by its :attr:`StageData.id`.

        :return: List of stages in ascending order of stage id.

        :raises FileNotFoundError: If a stage directory listed in the stage
            history does not exist.

        """
        if not hasattr(self, "_stages"):
            # Get the stages from the specified directories and load them
            self._stages = []
            for stage_dir in self._stage_history["relative_save_path"]:
                # Add the relative path to the stage directory
                stage_path = os.path.join(self.dir_path, stage_dir)

                # Load the stage data
                self._stages.append(
                    StageData(
                        stage_path, stage_attribute_loader=self._stage_attribute_loader
                    )
                )

            # Sort with the stage id
            self._stages.sort(key=lambda s: s.id)

        return self._stages

    @property
    def n_stages(self) -> int:
        """Number of stages that where performed."""
        return len(self._stage_history["ID"])

    @staticmethod
    def is_run_dir(dir_path: str | os.PathLike) -> tuple[bool, str]:
        """Checks if the provided path can be loaded with RunData.

        :returns: The boolean value if it is a run dir and an explaination why
            not.
        """
        if not os.path.isdir(dir_path):
            return False, "Provided path is not a dir."

        try:
            info_dict = load_dict_from_file(dir_path, "info.json")
        except AssertionError:
            return False, "File info.json could not be loaded. Not a run dir."

        if "stage_history" not in info_dict:
            return False, "No `stage_history` entry found in info.json. Not a run dir."

        return True, ""


def get_best_individual(
    run_data: RunData, global_statistics_row: int = 0, invert: bool = False
) -> tuple[Individual, float, dict[str, Any]]:
    """Return the individual with the best value for a global statistic.

    The statistic is selected by its row in ``run_data.global_statistics``.
    Use :meth:`get_analysis_selection_table` with analysis type
    ``"structures"`` to list the available statistic indices and names.

    :param run_data: The run data containing structures, global statistics,
        and key-value pairs.
    :type run_data: RunData
    :param global_statistics_row: Row index in ``run_data.global_statistics``
        that selects the statistic used for ranking. Defaults to ``0``.
    :type global_statistics_row: int
    :param invert: If ``True``, select the individual with the lowest
        statistic value instead of the highest. Defaults to ``False``.
    :type invert: bool
    :return: A tuple containing:

        - The best individual as an :class:`Individual` instance.
        - The statistic value used for ranking.
        - The key-value pairs associated with the best individual.

    :rtype: tuple[Individual, float, dict[str, Any]]
    """
    stat_values = []
    selected_key = run_data.global_statistics.loc[global_statistics_row, "names"]

    assert all(
        selected_key in kvp.keys() for kvp in run_data.key_value_pairs
    ), f"Key {selected_key} (row: {global_statistics_row}) not in one or multible key value pairs of structures db."

    stat_values = [
        key_value_pair[selected_key] for key_value_pair in run_data.key_value_pairs
    ]

    if invert:
        best_index = np.argmin(stat_values)
    else:
        best_index = np.argmax(stat_values)

    return (
        Individual.from_ase(run_data.structures[best_index]),
        stat_values[best_index],
        run_data.key_value_pairs[best_index],
    )


def get_run_overview(run_data: RunData) -> pd.Series:
    """Build an overview of a run as a :class:`pandas.Series`.

    The resulting series contains basic information to identify a run, such as
    the run name, description, number of stages, total generations, and total
    runtime.  Its index can be used as row labels when the series is printed or
    displayed as a table.

    :param run_data: The run data object providing the metadata.
    :type run_data: RunData
    :return: A series with an index labeling each entry.
    :rtype: pandas.Series
    """
    overview = pd.Series(
        {
            "name": run_data.name,
            "description": run_data.description,
            "n_stages": run_data.n_stages,
            "total_generations": run_data.n_generations,
            "total_runtime": run_data.total_runtime,
        },
    )
    return overview
