# Here I perform an example run with the cli
# From atomic structure to descriptor and then back to atomic structure

import os
from pathlib import PosixPath

import pytest
from click.testing import CliRunner

from fucrimodo.cli.main import cli


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
    run_config = fucrimodo_lab / "configs" / "run" / "test_run_config.py"
    save_dir = fucrimodo_lab / "data" / "results"

    results = runner.invoke(
        cli,
        [
            "run",
            "-p",
            1,
            "-c",
            str(run_config),
            "-s",
            str(save_dir),
            "-n",
            "test_run",
            str(target_file_path),
        ],
    )
    assert results.exit_code == 0, results.output

    # Analyse the run
    run_dir = save_dir / "test_run"
    analyse_config = fucrimodo_lab / "configs" / "analyse" / "run" / "default.py"
    results = runner.invoke(
        cli, ["analyse", "run", "-c", str(analyse_config), str(run_dir)]
    )
    assert results.exit_code == 0, results.output

    # Create global stats plot
    results = runner.invoke(
        cli,
        [
            "analyse",
            "run",
            "-c",
            str(analyse_config),
            "-r",
            "0",
            "-s",
            str(run_dir),
            str(run_dir),
        ],
    )
    assert results.exit_code == 0, results.output
    assert os.path.isfile(run_dir / "global_statistic_0.png")

    # Analyse stages
    analyse_config = fucrimodo_lab / "configs" / "analyse" / "stage" / "default.py"
    results = runner.invoke(
        cli,
        [
            "analyse",
            "stage",
            "-c",
            str(analyse_config),
            str(run_dir / "stage_1"),
        ],
    )
    assert results.exit_code == 0, results.output

    results = runner.invoke(
        cli,
        [
            "analyse",
            "stage",
            "-r",
            "0",
            "-s",
            str(run_dir / "stage_1"),
            "-c",
            str(analyse_config),
            str(run_dir / "stage_1"),
        ],
    )
    assert results.exit_code == 0, results.output
    assert os.path.isfile(run_dir / "stage_1" / "stage_statistic_0.png")
