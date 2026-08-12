import pytest
from fucrimodo.analysis.run_analysis import RunData


class TestRunData:
    @pytest.fixture
    def run_data(self, run_data_path):
        return RunData(run_data_path)

    def test_init(self, run_data: RunData):
        assert len(run_data.structures) == 30
        assert len(run_data.key_value_pairs) == len(run_data.structures)

        assert hasattr(run_data, "start_time_ms"), "Newest API assigns this"
