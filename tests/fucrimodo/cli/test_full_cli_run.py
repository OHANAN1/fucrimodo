# Here I perform an example run with the cli
# From atomic structure to descriptor and then back to atomic structure

from fucrimodo.cli.main import cli
from pathlib import PosixPath
from click.testing import CliRunner
import pytest


@pytest.fixture(scope="session")
def runner():
    return CliRunner()


@pytest.fixture(scope="session")
def fucrimodo_lab(runner, tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("original_dir")
    result = runner.invoke(cli, ["init", "-y", "-s", save_dir])
    assert result.exit_code == 0
    return save_dir / "fucrimodo_lab"


@pytest.fixture(scope="session")
def target_file_path(runner, fucrimodo_lab):
    atoms_path = fucrimodo_lab / "data" / "raw" / "test-target.xyz"
    save_path = fucrimodo_lab / "data" / "raw" / "test-target.json"
    config_path = fucrimodo_lab / "configs" / "utils" / "create_target_file_data.py"
    results = runner.invoke(
        cli,
        [
            "utils",
            "-c",
            config_path,
            "-a",
            f"atoms_path={atoms_path}",
            "-a",
            f"save_path={save_path}",
        ],
    )
    assert results.exit_code == 0

    return save_path


def test_fucrimodo_lab_init(fucrimodo_lab):
    assert type(fucrimodo_lab) is PosixPath


def test_structure_to_target_file_convertion(target_file_path):
    assert type(target_file_path) is PosixPath


@pytest.mark.slow
def test_run(runner, fucrimodo_lab, target_file_path):
    config_path = fucrimodo_lab / "configs" / "run" / "test_run_config.py"
    save_dir = fucrimodo_lab / "data" / "results"

    results = runner.invoke(
        cli,
        [
            "run",
            "-p",
            1,
            "-c",
            str(config_path),
            "-s",
            str(save_dir),
            str(target_file_path),
        ],
    )
    assert results.exit_code == 0, results.output
