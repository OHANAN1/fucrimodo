import os
from fucrimodo.analysis.stage_analysis import StageData
from fucrimodo.core.individual import Individual
import pytest
from fucrimodo.analysis.run_analysis import (
    RunData,
    get_best_individual,
    get_run_overview,
)


@pytest.fixture
def run_data(run_data_path):
    return RunData(run_data_path)


@pytest.mark.slow
class TestRunData:
    def test_init(self, run_data: RunData):
        assert hasattr(run_data, "_info_dict")
        assert hasattr(run_data, "_stage_history")

        assert len(run_data.structures) == 30
        assert len(run_data.key_value_pairs) == len(run_data.structures)

        start_time_ms = int(run_data.start_time.timestamp() * 1000)
        end_time_ms = int(run_data.end_time.timestamp() * 1000)

        assert 0 < start_time_ms
        assert start_time_ms < end_time_ms
        assert run_data.total_runtime_ms == pytest.approx(
            (run_data.end_time - run_data.start_time).total_seconds() * 1000, abs=10
        ), "Should work later tho"

        assert os.path.isfile(os.path.join(run_data.dir_path, "input_file.json"))

    def test_global_statistics(self, run_data: RunData):
        statistics = run_data.global_statistics.loc[0, "results"]

        assert "gen" in statistics
        assert "min" in statistics
        assert "max" in statistics
        assert "avg" in statistics
        assert "std" in statistics
        assert "stage_id" in statistics

        assert statistics["stage_id"].iloc[-1] == run_data.n_stages

        # This is because the statistics are not always recorded, when the
        # generation number changed
        assert len(statistics) <= run_data.n_generations

    def test_stages(self, run_data: RunData):
        assert run_data._info_dict["stage_history"]["ID"] == [1, 2, 3]
        assert run_data._info_dict["stage_history"]["relative_save_path"] == [
            "stage_1",
            "stage_2",
            "stage_3",
        ]

        assert all(type(s) is StageData for s in run_data.stages)
        assert len(run_data.stages) == run_data.n_stages

        assert [s.id for s in run_data.stages] == [1, 2, 3]


@pytest.mark.slow
def test_get_best_structures_tuple(run_data: RunData):
    struct, stats_val, key_value_pairs = get_best_individual(
        run_data=run_data, global_statistics_row=0, invert=False
    )

    assert isinstance(struct, Individual)
    assert stats_val == pytest.approx(
        run_data.global_statistics.loc[0, "results"]["max"].max()
    )
    assert key_value_pairs["Reference_Similarity"] == stats_val


@pytest.mark.slow
def test_get_run_overview(run_data: RunData):
    overview = get_run_overview(run_data)
    assert "name" in overview
    assert "description" in overview
    assert "n_stages" in overview
    assert "total_generations" in overview
    assert "total_runtime" in overview

    assert overview["n_stages"] == 3
