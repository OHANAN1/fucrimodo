import json
import os
from datetime import datetime

import pandas as pd


def load_dict_from_file(dir: str | os.PathLike, file_name: str) -> dict:
    """Load a dictionary from a json file with name :data:`file_name` from
    the directory ``dir``.

    :param file_name: Name of the file that should be loaded.

    :raises FileNotFoundError: If the file does not exist in the given
        directory.

    :return: The loaded dictionary.
    """
    file_path = os.path.join(dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_name} does not exist in {dir}.")

    with open(file_path, "r") as f:
        data = json.load(f)

    assert type(data) is dict, f"Loaded json is not a dict!"

    return data


def get_statistics_overview(stats: pd.DataFrame) -> pd.DataFrame:
    """Return an overview table with names, overall max, and overall min.

    If ``stats`` contains a ``titles`` column, it is included in the overview
    together with ``names``. For each row, the overall maximum and minimum are
    computed from the ``max`` and ``min`` columns of the DataFrame stored in
    ``results``.

    :param stats: DataFrame containing at least ``names`` and ``results``.
        Optionally contains ``titles``. Each value in ``results`` must be a
        DataFrame with ``max`` and ``min`` columns.
    :type stats: pandas.DataFrame
    :return: Overview DataFrame with columns ``names``, ``max``, and ``min``.
        If ``titles`` was present in ``stats``, it is included as well.
    :rtype: pandas.DataFrame
    :raises AssertionError: If the created overview is not a DataFrame.
    """
    if "titles" in stats:
        overview = stats[["titles", "names"]].copy()
    else:
        overview = stats[["names"]].copy()

    assert isinstance(overview, pd.DataFrame), "Provided stats are not a Dataframe."
    overview["max"] = [r["max"].max() for r in stats["results"]]
    overview["min"] = [r["min"].min() for r in stats["results"]]
    return overview


def get_start_time_from_info(info_dict: dict):
    """Extract the start time from an info dictionary.

    The dictionary is expected to contain either ``start_time_ms`` (a Unix
    timestamp in milliseconds) or ``start_time`` (a string formatted as
    ``"%Y-%m-%d %H:%M:%S"``). If both keys are present, ``start_time_ms`` is
    used.

    :param info_dict: Dictionary containing start-time information.
    :type info_dict: dict
    :return: The parsed start time.
    :rtype: datetime.datetime
    :raises AssertionError: If ``info_dict`` is not a ``dict``.
    :raises KeyError: If neither ``start_time_ms`` nor ``start_time`` is present.
    """
    assert isinstance(info_dict, dict)

    if "start_time_ms" in info_dict:
        return datetime.fromtimestamp(info_dict["start_time_ms"] / 1000.0)
    elif "start_time" in info_dict:
        return datetime.strptime(info_dict["start_time"], "%Y-%m-%d %H:%M:%S")
    else:
        raise KeyError(f"No time found in info dict: {info_dict}")


def get_end_time_from_info(info_dict: dict):
    """Extract the end time from an info dictionary.

    The dictionary is expected to contain either ``end_time_ms`` (a Unix
    timestamp in milliseconds) or ``end_time`` (a string formatted as
    ``"%Y-%m-%d %H:%M:%S"``). If both keys are present, ``end_time_ms`` is
    used.

    :param info_dict: Dictionary containing end-time information.
    :type info_dict: dict
    :return: The parsed end time.
    :rtype: datetime.datetime
    :raises AssertionError: If ``info_dict`` is not a ``dict``.
    :raises KeyError: If neither ``end_time_ms`` nor ``end_time`` is present.
    """
    assert isinstance(info_dict, dict)

    if "end_time_ms" in info_dict:
        return datetime.fromtimestamp(info_dict["end_time_ms"] / 1000.0)
    elif "end_time" in info_dict:
        return datetime.strptime(info_dict["end_time"], "%Y-%m-%d %H:%M:%S")
    else:
        raise KeyError(f"No time found in info dict: {info_dict}")
