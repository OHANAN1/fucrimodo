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
        if self.analysis_object == "run":
            self._analyse_run()
        elif self.analysis_object == "stage":
            self._analyse_stage()
        elif self.analysis_object == "multi_run":
            self._analyse_multi_run()
        else:
            raise ValueError(
                "Provided analysis object not found, "
                "only 'notebook', 'run', 'stage', 'multi_run' are allowed."
            )

    def _analyse_run(self):
        """Analyse a single run."""
        self.config.run(
            run_dir=self.dir_path,
            verbose=self.verbose,
            row=self.row,
            show=self.show,
            save_dir=self.save_dir,
        )

    def _analyse_stage(self):
        """Analyse a stage."""
        self.config.run(
            stage_dir=self.dir_path,
            verbose=self.verbose,
            row=self.row,
            show=self.show,
            save_dir=self.save_dir,
        )

    def _analyse_multi_run(self):
        """Analyse multiple runs."""
        self.config.run(
            multi_run_dir=self.dir_path,
            verbose=self.verbose,
            row=self.row,
            show=self.show,
            save_dir=self.save_dir,
        )


@click.command()
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
    help="Save the analysis results to this directory instead of displaying them.",
)
@click.option(
    "-r",
    "--row",
    type=int,
    help="Row index to analyse. If not provided, all rows are analysed.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a custom analysis configuration script. "
        "Default configs are located under configs/analysis/<object>/default.py."
    ),
)
def cli(analysis_object, dir_path, verbose, save_dir, row, config):
    """Analyse collected run or stage data.

    ANALYSIS_OBJECT is one of: notebook, run, stage, multi_run.

    DIR_PATH is the directory where the results of the run or stage are saved.
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
