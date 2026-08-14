import os
import click
from ..utils.import_helper import ConfigScript


class Runner:
    """Coordinate a single execution run.

    :param input_file_path: Path to the input/target file to process.
    :param verbose: Whether to enable verbose logging/output.
    :param save_dir: Directory where outputs are saved. If falsy, the
        current working directory is used.
    :param name: Name of the run.
    :param n_parallel: Number of parallel workers/processes to use.
    :param config_path: Path to the configuration script/file used to
        build the run configuration.

    The constructor creates ``save_dir`` if it does not already exist.
    """

    def __init__(
        self,
        input_file_path: str,
        verbose: bool,
        save_dir: str,
        name: str,
        n_parallel: int,
        config_path: str,
    ):
        save_path = save_dir if save_dir else os.getcwd()
        try:
            os.mkdir(save_path)
        except FileExistsError:
            pass
        self.save_dir = save_path

        self.input_file_path = input_file_path
        self.verbose = verbose
        self.name = name
        self.n_parallel = n_parallel
        self.run_config = ConfigScript(config_path)

    def run(self):
        """Execute the run using the loaded configuration.

        Delegates execution to :meth:`ConfigScript.run` with the stored
        run parameters.
        """
        # Load the target file
        self.run_config.run(
            name=self.name,
            save_dir=self.save_dir,
            target_file_path=self.input_file_path,
            n_parallel=self.n_parallel,
            verbose=self.verbose,
        )


@click.command(help="Perform a multi stage search on the provided input file.")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True, help="More output.")
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    default=os.path.join("configs", "run", "default.py"),
    help="Path to the run configuration script.",
)
@click.option(
    "-s",
    "--save_dir",
    type=click.Path(file_okay=False),
    help="Directory where the run outputs are saved.",
)
@click.option("-n", "--name")
@click.option(
    "-p", "--parallel", type=click.IntRange(min=1), default=1, show_default=True
)
def cli(input_file, verbose, config, save_dir, name, parallel):
    runner = Runner(
        input_file_path=input_file,
        verbose=verbose,
        config_path=config,
        save_dir=save_dir,
        name=name,
        n_parallel=parallel,
    )
    runner.run()
