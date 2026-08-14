import os
import click
from ..utils.import_helper import ConfigScript


class Runner:
    """Execute a configured utility with parsed parameters.

    :param config_path: Path to the configuration script used to build
        the runner's configuration.
    :param verbose: Whether to enable verbose output.
    :param params: Tuple of ``"key=value"`` strings to pass as keyword
        arguments to the configuration script.
    """

    def __init__(self, config_path, verbose: bool, params: tuple):
        self.args = self._parse_params(params=params)
        self.verbose = verbose
        self.config_script = ConfigScript(config_path)

    def run(self):
        """Run the utility.

        Execution the :meth:`ConfigScript.run` with the parsed
        parameters and verbose flag.
        """
        self.config_script.run(**self.args, verbose=self.verbose)

    def _parse_params(self, params):
        """Convert a tuple of ``"key=value"`` strings into a dict.

        :param params: Tuple of parameter strings.
        :return: Mapping of parameter names to their string values.
        :raises click.ClickException: If any parameter is not in
            ``"key=value"`` format.
        """
        parsed = {}
        for p in params:
            if "=" not in p:
                raise click.ClickException(
                    f"Parameter must be key=value format, got: {p!r}"
                )
            key, value = p.split("=", 1)
            parsed[key] = value
        return parsed


@click.command(
    no_args_is_help=True,
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=os.path.join("configs", "utils", "create_target_file_data.py"),
    show_default=True,
    help="Path to the utility configuration script.",
)
@click.option(
    "-a",
    "--arg",
    multiple=True,
    metavar="KEY=VALUE",
    help="Parameter passed as key=value. Can be used multiple times.",
)
@click.option("-v", "--verbose", is_flag=True, help="More output.")
def cli(config, arg, verbose):
    """Execute a utility configuration script.

    Runs a configurable utility located at CONFIG, passing any
    KEY=VALUE arguments to its main() function.
    """
    runner = Runner(config_path=config, verbose=verbose, params=arg)
    runner.run()
