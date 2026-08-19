import os

import pytest

from fucrimodo.analysis.run_analysis import RunData
from fucrimodo.analysis.stage_analysis import StageData
from fucrimodo.analysis.utils import get_statistics_overview, load_dict_from_file


@pytest.mark.slow
def test_load_dict_from_file(run_data_path):
    load_dict_from_file(run_data_path, "info.json")
    load_dict_from_file(run_data_path, "global_statistics.json")

    # Fails if file not exists
    with pytest.raises(FileNotFoundError):
        load_dict_from_file(run_data_path, "info_2.json")


@pytest.mark.slow
def test_get_statistics_overview_applied_to_stage(run_data_path):
    stage_data_path = os.path.join(run_data_path, "stage_1")
    stage_data = StageData(
        dir_path=stage_data_path,
    )
    fit_overview = get_statistics_overview(stage_data.fitness_statistics)

    assert "titles" in fit_overview
    assert "names" in fit_overview
    assert "max" in fit_overview
    assert "min" in fit_overview


@pytest.mark.slow
def test_get_statistics_overview_applied_to_run(run_data_path):
    stage_data = RunData(
        dir_path=run_data_path,
    )
    fit_overview = get_statistics_overview(stage_data.global_statistics)

    assert "names" in fit_overview
    assert "max" in fit_overview
    assert "min" in fit_overview

    # titles only used in fitness statistics
    assert "titles" not in fit_overview
