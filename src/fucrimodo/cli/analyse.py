import os
import click
from ..utils.import_helper import ConfigScript


class Runner:
    """Analyse data collected during a run or stage.

    :param analysis_object: Type of object to analyse. One of:
        ``run``, ``stage``, ``multi_run``.
    :param dir_path: Directory where the run or stage results are saved.
    :param verbose: Whether to enable verbose output.
    :param save_dir: Directory to save the analysis results. If ``None``,
        results are displayed instead of saved.
    :param row: Row index to analyse. If ``None``, all rows are analysed.
    :param config_path: Path to a custom configuration script. If ``None``,
        the default configuration for the analysis object is used.
    """

    def __init__(
        self,
        analysis_object: str,
        dir_path: str,
        verbose: bool,
        save_dir: str | None,
        row: int | None,
        config_path: str | None,
    ):
        """Initialize the runner and load the configuration."""
        self.analysis_object = analysis_object
        self.dir_path = dir_path
        self.verbose = verbose
        self.save_dir = save_dir
        self.show = save_dir is None
        self.row = row

        if config_path is not None:
            self.config = ConfigScript(config_path)
        else:
            default_path = os.path.join(
                "configs", "analysis", analysis_object, "default.py"
            )
            self.config = ConfigScript(default_path)

    def run(self):
        """Run the selected analysis.

        Dispatches to the analysis method that matches
        :attr:`analysis_object`.
        """

        try:
            self.config.run(
                dir_path=self.dir_path,
                verbose=self.verbose,
                row=self.row,
                show=self.show,
                save_dir=self.save_dir,
            )
        except TypeError:
            # TODO: Add this error message to config.run
            click.ClickException(
                "Could not load the config."
                f"Check that the file at {self.config.path} contains a method called `main()` that accepts the arguments `dir_path`, `verbose`, `row`, `show`, `save_dir`."
            )


@click.command(
    no_args_is_help=True,
    epilog=("Example: fucrimodo analyse -r 0 stage data/results/example-run/stage_1"),
)
@click.argument(
    "analysis_object",
    type=click.Choice(["run", "stage", "multi_run"]),
)
@click.argument(
    "dir_path",
    type=click.Path(exists=True, file_okay=False),
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
@click.option(
    "-s",
    "--save_dir",
    type=click.Path(exists=True, file_okay=False),
    help=(
        "Save analysis results to this directory instead of displaying them. "
        "The directory is created if it does not exist."
    ),
)
@click.option(
    "-r",
    "--row",
    type=int,
    help="Analyse only this row index of the statistics (default: only overview over all stats).",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a custom analysis configuration script. "
        "Defaults to configs/analysis/<analysis_object>/default.py."
    ),
)
def cli(analysis_object, dir_path, verbose, save_dir, row, config):
    """Analyse run or stage data.

    ANALYSIS_OBJECT is the type of object to analyse: run, stage, or multi_run.

    DIR_PATH is the directory containing the saved run or stage results.
    """
    runner = Runner(
        analysis_object=analysis_object,
        dir_path=dir_path,
        verbose=verbose,
        save_dir=save_dir,
        row=row,
        config_path=config,
    )
    runner.run()
