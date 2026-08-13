import pytest
import os

from fucrimodo.analysis.stage_analysis import (
    StageData,
    get_modification_overview,
    get_stage_overview,
)
from fucrimodo.customs.ga_stage.analysis import load_ga_stage_attributes


@pytest.fixture
def stage_data(run_data_path):
    stage_data_path = os.path.join(run_data_path, "stage_1")
    return StageData(
        dir_path=stage_data_path,
        stage_attribute_loader={"GAStage": load_ga_stage_attributes},
    )


@pytest.mark.slow
class TestStageData:
    def test_wrong_init(self, run_data_path):
        stage_data_path = os.path.join(run_data_path, "stage_1")

        # Test that error is raised if ga stage is not known
        with pytest.raises(ValueError):
            StageData(
                dir_path=stage_data_path,
                stage_attribute_loader={"NOTGAStage": load_ga_stage_attributes},
            )

    def test_params(self, stage_data: StageData):
        assert hasattr(stage_data, "_info_dict")
        assert stage_data.id == 1

        start_time_ms = int(stage_data.start_time.timestamp() * 1000)
        end_time_ms = int(stage_data.end_time.timestamp() * 1000)

        assert 0 < start_time_ms
        assert start_time_ms < end_time_ms
        assert stage_data.total_runtime_ms == pytest.approx(
            end_time_ms - start_time_ms, abs=10
        )

        assert stage_data.stage_attributes["parent_selection"]
        assert stage_data.stage_attributes["survivor_selection"]
        assert stage_data.stage_attributes["break_condition"]
        assert stage_data.n_generations

    def test_fitness(self, stage_data: StageData):
        statistics = stage_data.fitness_statistics.loc[0, "results"]
        statistics_name = stage_data.fitness_statistics.loc[0, "names"]
        assert statistics_name
        statistics_title = stage_data.fitness_statistics.loc[0, "titles"]
        assert statistics_title

        assert (
            len(statistics) == stage_data.n_generations + 1
        )  # +1 since initial population is also evaluated
        assert "gen" in statistics
        assert "min" in statistics
        assert "max" in statistics
        assert "avg" in statistics
        assert "std" in statistics

    def test_mutations(self, stage_data: StageData):
        mut_data = stage_data.stage_attributes["mutations"]

        assert "names" in mut_data
        assert "results" in mut_data
        assert "reprs" in mut_data

    def test_crossovers(self, stage_data: StageData):
        cross_data = stage_data.stage_attributes["crossovers"]

        assert "names" in cross_data
        assert "results" in cross_data
        assert "reprs" in cross_data


@pytest.mark.slow
def test_get_modification_overview(stage_data):
    mod_overview = get_modification_overview(
        stage_data=stage_data, modification_type="Mutation"
    )

    assert "total_calls" in mod_overview
    assert "total_fails" in mod_overview
    assert "total_survivors" in mod_overview
    assert "survivor_rate" in mod_overview
    assert "failed_rate" in mod_overview

    cross_overview = get_modification_overview(
        stage_data=stage_data, modification_type="Crossover"
    )

    assert "total_calls" in cross_overview
    assert "total_fails" in cross_overview
    assert "total_survivors" in cross_overview
    assert "survivor_rate" in cross_overview
    assert "failed_rate" in cross_overview


@pytest.mark.slow
def test_get_stage_overview(stage_data: StageData):
    stage_overview = get_stage_overview(stage_data=stage_data)

    assert "Name" in stage_overview
    assert "Description" in stage_overview
    assert "Type" in stage_overview
    assert "N_generations" in stage_overview
    assert "Parent Selection" in stage_overview
    assert "Survivor Selection" in stage_overview
    assert "Break Condition" in stage_overview
    assert "Parent Ratio" in stage_overview
    assert "N_fit" in stage_overview
    assert "N_mut" in stage_overview
    assert "N_cross" in stage_overview
    assert "Total Runtime" in stage_overview
