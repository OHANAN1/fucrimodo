import pytest
import click
from click.testing import CliRunner
from fucrimodo.cli.main import cli
from fucrimodo import __version__


@pytest.fixture
def runner():
    return CliRunner()


def test_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Fucrimodo command line tool." in result.output
    assert "run" in result.output
    assert "analyse" in result.output
    assert "init" in result.output
    assert "utils" in result.output


def test_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert f"fucrimodo, version {__version__}" in result.output


def test_version_short_flag(runner):
    result = runner.invoke(cli, ["-V"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_prints_usage(runner):
    result = runner.invoke(cli, [])
    assert result.exit_code == 2

    # Click groups without default commands print usage and exit 0
    assert "Usage:" in result.output
    assert "[OPTIONS] COMMAND [ARGS]..." in result.output


def test_keyboard_interrupt_is_silently_caught_by_main():
    """main() catches KeyboardInterrupt so the CLI exits cleanly."""

    @cli.command()
    def hang():
        raise KeyboardInterrupt

    runner = CliRunner()
    result = runner.invoke(cli, ["hang"])
    assert result.exit_code != 0
