import pytest
import os
from pathlib import Path
import shutil
import json

from fucrimodo.analysis.multi_run_analysis import (
    MultiRunData,
    get_all_global_statistics_overview,
    get_multi_run_overview,
)


@pytest.fixture
def multi_run_data(run_data_path):
    run_dir_original = Path(run_data_path)
    save_dir = run_dir_original.parent
    run_dir_copy = save_dir / "run_dir_copy"

    if not os.path.isdir(run_dir_copy):
        shutil.copytree(run_dir_original, run_dir_copy, dirs_exist_ok=True)

        info_json_path = run_dir_copy / "info.json"
        # Overwrite the name attr in the info.json file with another name
        with open(info_json_path, "r") as f:
            info_dict = json.load(f)
        info_dict["name"] = "test_run_copy"
        with info_json_path.open("w", encoding="utf-8") as f:
            json.dump(info_dict, f)

        # Also move random file to the multi_stage_dir, to test if the MultiStageData
        # successfully ignores it when loading data
        random_file_path = os.path.join("data", "random_file")
        shutil.copy(random_file_path, run_dir_copy)

    return MultiRunData(save_dir)


@pytest.mark.slow
class TestMultiRunData:
    def test_loading_runs(self, multi_run_data: MultiRunData):
        assert len(multi_run_data.runs) == 2
        assert "test_run_copy" in [run.name for run in multi_run_data.runs]
        assert "test_run" in [run.name for run in multi_run_data.runs]

    def test_total_run_time(self, multi_run_data: MultiRunData):

        # Since run is only a copy, the total run time should be the same as the
        # total run time of one run
        assert multi_run_data.total_runtime_ms == pytest.approx(
            multi_run_data.runs[0].total_runtime_ms, abs=10
        )


@pytest.mark.slow
def test_get_all_global_statistics_overview(multi_run_data: MultiRunData):
    overview = get_all_global_statistics_overview(multi_run_data=multi_run_data)

    assert len(overview) == 2
    assert "run" in overview
    assert "Reference_Similarity_min_max" in overview
    assert "Volume_min_max" in overview


@pytest.mark.slow
def test_get_multi_run_overview(multi_run_data: MultiRunData):
    overview = get_multi_run_overview(multi_run_data=multi_run_data)

    assert len(overview) == 2
    assert "names" in overview
    assert "descriptions" in overview
    assert "n_stages" in overview
    assert "total_generations" in overview
    assert "total_runtime" in overview
