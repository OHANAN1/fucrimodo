import importlib
import click
from fucrimodo import __version__


@click.group(
    help="Fucrimodo command line tool.",
    no_args_is_help=True,
)
@click.version_option(
    __version__,
    "-V",
    "--version",
    prog_name="fucrimodo",
)
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)


commands = [
    ("run", "fucrimodo.cli.run"),
    ("analyse", "fucrimodo.cli.analyse"),
    ("init", "fucrimodo.cli.init"),
    ("utils", "fucrimodo.cli.utils"),
]

for name, module_name in commands:
    module = importlib.import_module(module_name)
    cli.add_command(module.cli, name=name)


def main():
    try:
        cli()
    except KeyboardInterrupt:
        pass
