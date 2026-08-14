import importlib
import click
from fucrimodo import __version__


@click.group(
    no_args_is_help=True,
    invoke_without_command=False,
    epilog=(
        "Run 'fucrimodo COMMAND --help' for more information on a specific command.\n\n"
        "\b\n"
        "Example:\n"
        "  fucrimodo init --save_dir ./my_experiments\n"
        "  fucrimodo run -s 'data' -n 'run_01' ./data/raw/example_target_file.json \n"
        "  fucrimodo analyse run ./data/run_01"
    ),
)
@click.version_option(
    __version__,
    "-V",
    "--version",
    prog_name="fucrimodo",
)
@click.pass_context
def cli(ctx):
    """Fucrimodo command line tool.

    Fucrimodo helps configure, run, store, and analyse reproducible
    multi-stage optimization experiments.

    Available commands:

    \b
      init      Create a new fucrimodo_lab directory with default configs.
      run       Run an optimization experiment or stage.
      analyse   Analyse data collected during a run or stage.
      utils     Utility commands for managing fucrimodo projects.

    To start please read the tutorials in the docs or if you feel brave just
    create a new fucrimodo_lab with 'fucrimodo init'.

    """
    ctx.ensure_object(dict)


# Subcommands are registered from their respective modules.
# Each module must expose a `cli` click command.
_COMMANDS: dict[str, str] = {
    "run": "fucrimodo.cli.run",
    "analyse": "fucrimodo.cli.analyse",
    "init": "fucrimodo.cli.init",
    "utils": "fucrimodo.cli.utils",
}

for name, module_name in _COMMANDS.items():
    module = importlib.import_module(module_name)
    cli.add_command(module.cli, name=name)


def main():
    try:
        cli()
    except KeyboardInterrupt:
        pass
