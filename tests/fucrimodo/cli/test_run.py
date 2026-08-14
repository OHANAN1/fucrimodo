import os
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from fucrimodo.cli.run import Runner, cli


class TestRunner:
    @patch("fucrimodo.cli.run.ConfigScript")
    def test_init_uses_custom_config_when_given(self, mock_config_script, tmp_path):
        mock_config = MagicMock()
        mock_config_script.return_value = mock_config

        save_dir = tmp_path / "out"
        runner = Runner(
            input_file_path="/some/input.txt",
            verbose=True,
            save_dir=str(save_dir),
            name="test_run",
            n_parallel=4,
            config_path="/custom/config.py",
        )

        mock_config_script.assert_called_once_with("/custom/config.py")
        assert runner.input_file_path == "/some/input.txt"
        assert runner.verbose is True
        assert runner.save_dir == str(save_dir)
        assert runner.name == "test_run"
        assert runner.n_parallel == 4
        assert runner.run_config is mock_config
        assert save_dir.exists()  # save_dir was created

    @patch("fucrimodo.cli.run.ConfigScript")
    def test_init_uses_cwd_when_save_dir_not_given(
        self, mock_config_script, tmp_path, monkeypatch
    ):
        mock_config = MagicMock()
        mock_config_script.return_value = mock_config

        monkeypatch.chdir(tmp_path)
        runner = Runner(
            input_file_path="/some/input.txt",
            verbose=False,
            save_dir=None,
            name="default_run",
            n_parallel=1,
            config_path="/custom/config.py",
        )

        assert runner.save_dir == str(tmp_path)
        mock_config_script.assert_called_once_with("/custom/config.py")

    @patch("fucrimodo.cli.run.ConfigScript")
    def test_init_creates_save_dir_when_it_does_not_exist(
        self, mock_config_script, tmp_path
    ):
        mock_config = MagicMock()
        mock_config_script.return_value = mock_config

        save_dir = tmp_path / "new_output_dir"
        assert not save_dir.exists()

        runner = Runner(
            input_file_path="/some/input.txt",
            verbose=False,
            save_dir=str(save_dir),
            name="test",
            n_parallel=1,
            config_path="/custom/config.py",
        )

        assert runner.save_dir == str(save_dir)
        assert save_dir.exists()

    @patch("fucrimodo.cli.run.ConfigScript")
    def test_run_dispatches_to_config(self, mock_config_script, tmp_path):
        mock_config = MagicMock()
        mock_config_script.return_value = mock_config

        save_dir = tmp_path / "out"
        runner = Runner(
            input_file_path="/some/input.txt",
            verbose=True,
            save_dir=str(save_dir),
            name="test_run",
            n_parallel=4,
            config_path="/custom/config.py",
        )
        runner.run()

        mock_config.run.assert_called_once_with(
            name="test_run",
            save_dir=str(save_dir),
            target_file_path="/some/input.txt",
            n_parallel=4,
            verbose=True,
        )


class TestCli:
    @pytest.fixture
    def dummy_config(self, tmp_path):
        config = tmp_path / "dummy_config.py"
        config.write_text("# dummy config\n")
        return str(config)

    @pytest.fixture
    def input_file(self, tmp_path):
        input_file = tmp_path / "input.txt"
        input_file.write_text("dummy input\n")
        return str(input_file)

    @patch("fucrimodo.cli.run.Runner")
    def test_cli_basic(self, mock_runner_class, dummy_config, input_file):
        runner_instance = MagicMock()
        mock_runner_class.return_value = runner_instance

        result = CliRunner().invoke(
            cli,
            [
                input_file,
                "-c",
                dummy_config,
            ],
        )

        assert result.exit_code == 0
        mock_runner_class.assert_called_once_with(
            input_file_path=input_file,
            verbose=False,
            save_dir=None,
            name=None,
            n_parallel=1,
            config_path=dummy_config,
        )
        runner_instance.run.assert_called_once()

    @patch("fucrimodo.cli.run.Runner")
    def test_cli_uses_default_config_path(
        self, mock_runner_class, input_file, tmp_path, monkeypatch
    ):
        runner_instance = MagicMock()
        mock_runner_class.return_value = runner_instance

        # Create the default config file expected by the CLI
        default_config = tmp_path / "configs" / "run" / "default.py"
        default_config.parent.mkdir(parents=True)
        default_config.write_text("# default config\n")

        # Change cwd so the relative default path resolves to the file above
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(cli, [input_file])

        assert result.exit_code == 0
        expected_config = os.path.join("configs", "run", "default.py")
        mock_runner_class.assert_called_once_with(
            input_file_path=input_file,
            verbose=False,
            save_dir=None,
            name=None,
            n_parallel=1,
            config_path=expected_config,
        )
        runner_instance.run.assert_called_once()

    @patch("fucrimodo.cli.run.Runner")
    def test_cli_with_all_options(
        self, mock_runner_class, dummy_config, input_file, tmp_path
    ):
        runner_instance = MagicMock()
        mock_runner_class.return_value = runner_instance

        save_dir = tmp_path / "out"
        result = CliRunner().invoke(
            cli,
            [
                input_file,
                "-v",
                "-c",
                dummy_config,
                "-s",
                str(save_dir),
                "-n",
                "my_run",
                "-p",
                "4",
            ],
        )

        assert result.exit_code == 0
        mock_runner_class.assert_called_once_with(
            input_file_path=input_file,
            verbose=True,
            save_dir=str(save_dir),
            name="my_run",
            n_parallel=4,
            config_path=dummy_config,
        )
        runner_instance.run.assert_called_once()

    def test_cli_rejects_nonexistent_input_file(self, dummy_config):
        result = CliRunner().invoke(
            cli,
            [
                "does_not_exist.txt",
                "-c",
                dummy_config,
            ],
        )
        assert result.exit_code != 0
        assert "does_not_exist.txt" in result.output

    def test_cli_rejects_parallel_less_than_one(self, input_file, dummy_config):
        result = CliRunner().invoke(
            cli,
            [
                input_file,
                "-p",
                "0",
                "-c",
                dummy_config,
            ],
        )
        assert result.exit_code != 0
        assert "Invalid value" in result.output

    def test_cli_requires_input_file(self, dummy_config):
        result = CliRunner().invoke(cli, ["-c", dummy_config])
        assert result.exit_code != 0
        assert "Missing argument" in result.output
