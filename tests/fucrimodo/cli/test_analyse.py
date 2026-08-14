import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from fucrimodo.cli.analyse import Runner, cli


class TestRunner:
    @patch("fucrimodo.cli.analyse.ConfigScript")
    def test_init_uses_custom_config_when_given(self, mock_config_script):
        mock_config = MagicMock()
        mock_config_script.return_value = mock_config

        runner = Runner(
            analysis_object="run",
            dir_path="/some/dir",
            verbose=True,
            save_dir=None,
            row=None,
            config_path="/custom/config.py",
        )

        mock_config_script.assert_called_once_with("/custom/config.py")
        assert runner.analysis_object == "run"
        assert runner.dir_path == "/some/dir"
        assert runner.verbose is True
        assert runner.save_dir is None
        assert runner.show is True  # because save_dir is None
        assert runner.row is None
        assert runner.config is mock_config

    @patch("fucrimodo.cli.analyse.ConfigScript")
    def test_init_uses_default_config_when_no_custom_path(self, mock_config_script):
        mock_config = MagicMock()
        mock_config_script.return_value = mock_config

        runner = Runner(
            analysis_object="stage",
            dir_path="/some/dir",
            verbose=False,
            save_dir="/save/dir",
            row=5,
            config_path=None,
        )

        expected_path = os.path.join("configs", "analysis", "stage", "default.py")
        mock_config_script.assert_called_once_with(expected_path)
        assert runner.show is False  # because save_dir is provided
        assert runner.row == 5

    @patch("fucrimodo.cli.analyse.ConfigScript")
    def test_run_dispatches_to_config(self, mock_config_script):
        mock_config = MagicMock()
        mock_config_script.return_value = mock_config

        runner = Runner(
            analysis_object="multi_run",
            dir_path="/some/dir",
            verbose=True,
            save_dir="/out/dir",
            row=2,
            config_path=None,
        )
        runner.run()

        mock_config.run.assert_called_once_with(
            dir_path="/some/dir",
            verbose=True,
            row=2,
            show=False,
            save_dir="/out/dir",
        )


class TestCli:
    @pytest.fixture
    def dummy_config(self, tmp_path):
        config = tmp_path / "dummy_config.py"
        config.write_text("# dummy config\n")
        return str(config)

    @patch("fucrimodo.cli.analyse.Runner")
    def test_cli_basic(self, mock_runner_class, dummy_config):
        runner_instance = MagicMock()
        mock_runner_class.return_value = runner_instance

        with CliRunner().isolated_filesystem():
            os.makedirs("data")
            result = CliRunner().invoke(cli, ["run", "data"])

        assert result.exit_code == 0
        mock_runner_class.assert_called_once_with(
            analysis_object="run",
            dir_path="data",
            verbose=False,
            save_dir=None,
            row=None,
            config_path=None,
        )
        runner_instance.run.assert_called_once()

    @patch("fucrimodo.cli.analyse.Runner")
    def test_cli_with_all_options(self, mock_runner_class, dummy_config):
        runner_instance = MagicMock()
        mock_runner_class.return_value = runner_instance

        with CliRunner().isolated_filesystem():
            os.makedirs("data")
            os.makedirs("out")
            result = CliRunner().invoke(
                cli,
                [
                    "stage",
                    "data",
                    "-v",
                    "-s",
                    "out",
                    "-r",
                    "3",
                    "-c",
                    dummy_config,
                ],
            )

        assert result.exit_code == 0
        mock_runner_class.assert_called_once_with(
            analysis_object="stage",
            dir_path="data",
            verbose=True,
            save_dir="out",
            row=3,
            config_path=dummy_config,
        )
        runner_instance.run.assert_called_once()

    def test_cli_rejects_invalid_analysis_object(self):
        with CliRunner().isolated_filesystem():
            os.makedirs("data")
            result = CliRunner().invoke(cli, ["notebook", "data"])

        assert result.exit_code != 0
        assert "Invalid value" in result.output

    def test_cli_requires_dir_path(self):
        result = CliRunner().invoke(cli, ["run"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output
