import pytest
from click.testing import CliRunner
from fucrimodo.cli.init import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Generate a fucrimodo_lab directory with default configs." in result.output


def test_run_init(runner, tmp_path):
    result = runner.invoke(cli, ["-s", f"{tmp_path}", "-y"])

    assert result.exit_code == 0
    lab_path = tmp_path / "fucrimodo_lab"

    assert (lab_path).exists()
    assert (lab_path / "README.md").exists()

    assert (lab_path / "configs").exists()
    assert (lab_path / "configs" / "run").exists()
    assert (lab_path / "configs" / "run" / "default.py").exists()
    assert (lab_path / "configs" / "run" / "test_run_config.py").exists()
    assert (lab_path / "configs" / "analyse").exists()
    assert (lab_path / "configs" / "analyse" / "stage" / "default.py").exists()
    assert (lab_path / "configs" / "analyse" / "run" / "default.py").exists()
    assert (lab_path / "configs" / "analyse" / "multi_run" / "default.py").exists()

    assert (lab_path / "data").exists()
    assert (lab_path / "data" / "raw").exists()
    assert (lab_path / "data" / "raw" / "example-target.xyz").exists()


def test_run_init_with_error(runner, tmp_path):
    # If provided dir does not exist
    result = runner.invoke(cli, ["-s", f"{tmp_path}/random_dir", "-y"])
    assert result.exit_code != 0
