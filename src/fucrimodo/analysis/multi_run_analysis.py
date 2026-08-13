import os
import pandas as pd
from typing import Callable

from ..customs.ga_stage.analysis import load_ga_stage_attributes
from .run_analysis import RunData
from .utils import get_statistics_overview


class MultiRunData:
    """Collects and structures the data produced by multiple runs.

    The provided directory is expected to contain one or more valid run
    subdirectories. Each subdirectory is checked with
    :meth:`RunData.is_run_dir` and, if valid, loaded as a :class:`RunData`
    instance.

    :param dir_path: Path to the directory containing the run
        directories.
    :param stage_attribute_loader: Mapping from stage type names to callables
        that load additional stage attributes. Each callable receives a stage
        name and a dictionary of stage info and returns a dictionary of
        attributes, see :class:`StageData` for more info.  Defaults to
        ``{"GAStage": load_ga_stage_attributes}``.

    :raises AssertionError: If no valid runs can be loaded from ``multi_run_dir``.
    """

    def __init__(
        self,
        dir_path: str | os.PathLike,
        stage_attribute_loader: dict[str, Callable[[str, dict], dict]] = {
            "GAStage": load_ga_stage_attributes
        },
    ):
        self._dir_path = dir_path
        self._stage_attribute_loader = stage_attribute_loader

        self._runs = self.__load_runs_from_dir(
            stage_attr_loader=self._stage_attribute_loader
        )
        assert len(self._runs) > 0, "No runs found."

    @property
    def dir_path(self) -> str | os.PathLike:
        """Path to the multi run directory."""
        return self._dir_path

    @property
    def runs(self) -> list[RunData]:
        """All runs successfully loaded from the subdirs of ``dir_path``.

        (`That was weird` ~ Sosuke [after several waves with eyes fail to catch him by the shore])

        :return: List of RunData objects.
        """
        if not hasattr(self, "_runs"):
            # Create a list of RunData objects
            self._runs = self.__load_runs_from_dir(
                stage_attr_loader=self._stage_attribute_loader
            )
            assert len(self._runs) > 0, "No runs found."
        return self._runs

    @property
    def total_runtime_ms(self) -> int:
        """Return the wall-clock span covered by all loaded runs.

        It is calulate by taking the time delta from the earliest start time to
        the latest end time across all runs. If runs are unfinished, the last
        stage is not counted, since the run info is updated only after a stage
        stopped.

        :returns: Time span in milliseconds.
        :rtype: int
        """
        if not hasattr(self, "_total_runtime_ms"):

            # Smallest start time and largest end time
            earliest_start = min(self.runs, key=lambda r: r.start_time).start_time
            latest_end = max(self.runs, key=lambda r: r.end_time).end_time

            # Total duration as timedelta
            total_duration = latest_end - earliest_start

            # Total duration in milliseconds
            self._total_runtime_ms = int(total_duration.total_seconds() * 1000)

        return self._total_runtime_ms

    @property
    def n_runs(self) -> int:
        """Return the number of loaded runs.

        :returns: Number of valid runs found in ``multi_run_dir``.
        :rtype: int
        """
        return len(self.runs)

    def __load_runs_from_dir(self, stage_attr_loader) -> list[RunData]:
        """Load all valid run directories inside :attr:`multi_run_dir`.

        Iterates over the entries in :attr:`multi_run_dir`, checks each with
        :meth:`RunData.is_run_dir`, and loads valid ones as :class:`RunData`
        instances.

        :param stage_attr_loader: Stage attribute loader mapping passed to each
            :class:`RunData` constructor.

        :returns: List of loaded run data objects.
        :rtype: list[RunData]
        """
        runs = []
        for potential_run_dir in os.listdir(self._dir_path):
            pot_run_dir = os.path.join(self._dir_path, potential_run_dir)
            is_run_dir, _ = RunData.is_run_dir(pot_run_dir)
            if is_run_dir:
                run = RunData(pot_run_dir, stage_attribute_loader=stage_attr_loader)
                runs.append(run)

        return runs


def get_multi_run_overview(multi_run_data: MultiRunData) -> pd.DataFrame:
    """Create an overview table of all loaded runs.

    Collects basic per-run information into a :class:`pandas.DataFrame` that
    can be printed or displayed as a table.

    :param multi_run_data: The multi-run data container holding the runs to
        summarize.
    :type multi_run_data: MultiRunData
    :returns: DataFrame with columns ``names``, ``description``, ``n_stages``,
        ``total_generations`` and ``total_runtime``.
    :rtype: pd.DataFrame
    """
    overview = pd.DataFrame(
        {
            "names": [run.name for run in multi_run_data.runs],
            "descriptions": [run.description for run in multi_run_data.runs],
            "n_stages": [run.n_stages for run in multi_run_data.runs],
            "total_generations": [run.n_generations for run in multi_run_data.runs],
            "total_runtime": [run.total_runtime for run in multi_run_data.runs],
        }
    )
    return overview


def get_all_global_statistics_overview(
    multi_run_data: MultiRunData,
    round_decimals: int = 5,
) -> pd.DataFrame:
    """Create an overview table of the global statistics for all runs.

    For each run the global statistics are retrieved and summarized as a
    ``min, max`` string per statistic. The resulting DataFrame has one row per
    run and one column per statistic plus a ``run`` column.

    :param multi_run_data: The multi-run data container holding the runs to
        summarize.
    :type multi_run_data: MultiRunData
    :param round_decimals: Number of decimal places to round the minimum and
        maximum values to. Defaults to ``5``.
    :type round_decimals: int
    :returns: DataFrame with a ``run`` column and one column per global
        statistic named ``<statistics-name>_min_max``.
    :rtype: pd.DataFrame
    """
    rows: list[dict[str, str]] = []

    for run_data in multi_run_data.runs:
        global_stats = get_statistics_overview(run_data.global_statistics)

        row: dict[str, str] = {"run": run_data.name}

        for name, min_val, max_val in zip(
            global_stats["names"], global_stats["min"], global_stats["max"]
        ):
            row[f"{name}_min_max"] = (
                f"{min_val:.{round_decimals}f}, {max_val:.{round_decimals}f}"
            )

        rows.append(row)

    return pd.DataFrame(rows)
