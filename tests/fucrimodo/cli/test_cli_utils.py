import os
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from fucrimodo.cli.utils import Runner, cli


class TestRunner:

    @patch("fucrimodo.cli.utils.ConfigScript")
    def test_parse_params_valid(
        self, _mock_config_script
    ):  # The _mock_config_script has to be here because of the patch
        runner = Runner("dummy_config.py", False, ("key1=value1", "key2=value2"))
        assert runner.args == {"key1": "value1", "key2": "value2"}

    @patch("fucrimodo.cli.utils.ConfigScript")
    def test_parse_params_empty(self, _mock_config_script):
        runner = Runner("dummy_config.py", False, ())
        assert runner.args == {}

    @patch("fucrimodo.cli.utils.ConfigScript")
    def test_parse_params_value_with_equals(self, _mock_config_script):
        runner = Runner("dummy_config.py", False, ("key=value=with=equals",))
        assert runner.args == {"key": "value=with=equals"}

    def test_parse_params_missing_equals_raises(self):
        with pytest.raises(
            click.ClickException, match="Parameter must be key=value format"
        ):
            Runner("dummy_config.py", False, ("badparam",))

    @patch("fucrimodo.cli.utils.ConfigScript")
    def test_run_calls_config_script(self, mock_config_script_class):
        mock_config_script = MagicMock()
        mock_config_script_class.return_value = mock_config_script

        runner = Runner("dummy_config.py", True, ("foo=bar", "baz=qux"))
        runner.run()

        mock_config_script_class.assert_called_once_with("dummy_config.py")
        mock_config_script.run.assert_called_once_with(
            foo="bar", baz="qux", verbose=True
        )


class TestCli:
    @pytest.fixture
    def dummy_config(self, tmp_path):
        config = tmp_path / "dummy_config.py"
        config.write_text("# dummy config\n")
        return str(config)

    @patch("fucrimodo.cli.utils.ConfigScript")
    def test_cli_runs_with_defaults(self, mock_config_script_class, dummy_config):
        mock_config_script = MagicMock()
        mock_config_script_class.return_value = mock_config_script

        runner = CliRunner()
        result = runner.invoke(cli, ["-c", dummy_config])

        assert result.exit_code == 0
        mock_config_script_class.assert_called_once_with(dummy_config)
        mock_config_script.run.assert_called_once_with(verbose=False)

    @patch("fucrimodo.cli.utils.ConfigScript")
    def test_cli_with_params_and_verbose(self, mock_config_script_class, dummy_config):
        mock_config_script = MagicMock()
        mock_config_script_class.return_value = mock_config_script

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["-c", dummy_config, "-a", "a=1", "-a", "b=2", "-v"],
        )

        assert result.exit_code == 0
        mock_config_script.run.assert_called_once_with(a="1", b="2", verbose=True)

    @patch("fucrimodo.cli.utils.ConfigScript")
    def test_cli_default_config_path(
        self, mock_config_script_class, tmp_path, monkeypatch
    ):
        """Test the default config path by creating the expected file."""
        # Create the default config directory and file
        default_dir = tmp_path / "configs" / "utils"
        default_dir.mkdir(parents=True)
        default_config = default_dir / "create_target_file_data.py"
        default_config.write_text("# default config\n")

        # Change working directory so the relative default path resolves
        monkeypatch.chdir(tmp_path)

        mock_config_script = MagicMock()
        mock_config_script_class.return_value = mock_config_script

        runner = CliRunner()
        result = runner.invoke(cli)

        assert result.exit_code == 0
        expected_default = os.path.join(
            "configs", "utils", "create_target_file_data.py"
        )
        mock_config_script_class.assert_called_once_with(expected_default)
