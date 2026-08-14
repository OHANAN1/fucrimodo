import datetime
import json
import logging
import os
from typing import Callable

from deap.tools.crossover import warnings
import numpy as np
from ase import db
from ase.db.core import Database
from deap import tools
from pathlib import Path
import shutil

from .individual import Individual
from .population import Population
from .utils.log_utils import setup_run_logger, setup_stage_logger

from .abstracts import Stage


class MultiStageSearch:
    """Class to run the multi-stage optimization algorithm.

    This class is used to manage data, time-keeping, organization of stages, ...
    during a multi-stage search.  On initialization the class creates a
    :attr:`run_dir` inside the :attr:`save_dir` where information about the run
    can be stored. A new directory inside :attr:`run_dir` is then automatically
    assigned for each stage so it can store its own data.
    After setting up the MultiStageSearch a stage can be run with the `run` method.

    :param save_dir: Directory where a dictionary should be created to store
        the data of the run.
    :param target_features: Array with the target features that the
        optimization algorithm should invert.
    :param descriptor_object: Descriptor object that is used to
        calculate the features of the individuals.
    :param descriptive_name: Optional name of the run. If no name is given,
        the current time and date is used. The directory, where all data is
        stored (:attr:`run_dir`) is initialized with the :attr:`name`.
    :param description: Optional description for the run. Will be stored
        automatically in the info.json in the :attr:`run_dir`.
    :param global_statistics_dict: Dictionary, where the keys are the
        names of the statistics and the values are functions that calculate the
        statistics for an individual. The statistics are calculated for each
        iteration that modifies the population (e.g. in Genetic Algorithms they
        are calculated for each generation) of all stages of the optimization
        algorithm. If not initialized at start must be set later.
    :param log_level: Log level of the global logger. Set it to logging.INFO to
        see the progress of the run.
    :param verbose: If set to True, the global logger also logs to the console
        in addition to the log file.
    """

    def __init__(
        self,
        save_dir: str | os.PathLike,
        target_features: np.ndarray,
        descriptor_object,
        descriptive_name: str | None = None,
        description: str = "",
        global_statistics_dict: dict[str, Callable[[Individual], float]] | None = None,
        log_level: int = logging.INFO,
        verbose: bool = True,
        n_jobs: int = 1,
    ) -> None:
        # If no descriptive name is given, use the current time and date.
        if descriptive_name is None:
            self._name = self.__get_time_string()
        else:
            self._name = descriptive_name

        self._description = description
        self._target_features = target_features
        self._descriptor_object = descriptor_object
        self.n_jobs = n_jobs
        self.max_number_of_parallel_jobs = n_jobs

        # Create the dictionary to store the data of the run
        self._run_dir = self.__create_run_dir(save_dir)

        # Save the global statistics dictionary to access it during saving
        self.global_statistics_dict = global_statistics_dict

        # Initially set the stage id to 0
        self.current_stage_id = 0

        # Set up the global logger for the run
        self.logger = setup_run_logger(
            log_file_path=f"{self.run_dir}/run.log",
            run_name=self.name,
            log_level=log_level,
            verbose=verbose,
        )

        # Initialize empty stage history
        self._stage_history = {
            "ID": [],
            "relative_save_path": [],
        }

        self.logger.info(f"Initialized run {self.name}")

        # Check node on which the script runs for debug purposes of run fails
        value = os.environ.get("SLURMD_NODENAME", "Not set")
        if value != "Not set":
            self.logger.info(f"Running on: {value}")
        else:
            self.logger.info("Not running on slurm.")

    @property
    def name(self) -> str:
        """Name of the run. The :attr:`run_dir` will be named after it."""
        # Define name attribute without setter, since it should never be changed
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @description.setter
    def description(self, value: str):
        self._description = value

    @property
    def log_level(self) -> int:
        return self.logger.level

    @log_level.setter
    def log_level(self, value: int):
        self._log_level = value

        # Update the log level of the global logger
        self.logger.setLevel(value)

    @property
    def start_time(self) -> datetime.datetime:
        """Time the run is started.

        Is set automatically when the `run` method is called for the first
        stage.
        """
        if not hasattr(self, "_start_time"):
            raise AttributeError(
                "No start time set. Please first run a stage before calling start_time."
            )
        return self._start_time

    @property
    def end_time(self) -> datetime.datetime:
        """Time the run ended.

        If not assigned yet, returns the :attr:`start_time`.
        """
        if not hasattr(self, "_end_time"):
            # If the end time is not set, return the start time
            return self.start_time
        return self._end_time

    def set_end_time(self):
        """Method to set the end time of the run to the current time."""
        self._end_time = datetime.datetime.now()

    @property
    def max_number_of_parallel_jobs(self) -> int:
        """Maximum number of parallel jobs that should be run.

        Only for information. The number of parallel jobs must be set
        correctly in the stages but this can be used to check if the number
        of parallel jobs is set correctly in all stages.
        """
        if not hasattr(self, "_max_number_of_parallel_jobs"):
            raise AttributeError(
                "Please set the maximum number of parallel jobs manually before calling."
            )
        return self._max_number_of_parallel_jobs

    @max_number_of_parallel_jobs.setter
    def max_number_of_parallel_jobs(self, value: int):
        self._max_number_of_parallel_jobs = value

    @property
    def global_statistics_dict(self) -> dict[str, Callable[[Individual], float]] | None:
        """Dictionary of the global statistics dictionary.

        The global statistics are a list of descriptive names for each statistic
        and a callable function that can evaluate individuals. E.g.:

        .. code-block::python

            global_statistics_dict = {
                "volume": lambda ind: ind.get_volume(),
            }

        Global statistics are tracked for all stages and are stored in the file
        :attr:`run_dir`/global_statistics.json.

        Setting a new global statistics dict also creates the :attr:`_global_statistics` and
        the :attr:`global_logbook` for the new statistics.
        A new global statistics dict can only be set if no records have been made yet.
        """
        if not hasattr(self, "_global_statistics_dict"):
            self._global_statistics_dict = None
        return self._global_statistics_dict

    @global_statistics_dict.setter
    def global_statistics_dict(
        self, value: dict[str, Callable[[Individual], float]] | None
    ):
        # Only allow setting new statistics if no stats have been recorded yet.
        assert len(self.global_logbook) == 0

        self._global_statistics_dict = value

        # Create the global statistics and the logbook for the new statistics
        if value is None:
            self._global_statistics = None
            self._global_log = self._get_global_logbook()
        else:
            self._global_statistics = self.__create_global_statistics(
                self._global_statistics_dict
            )
            self._global_log = self._get_global_logbook()

    @property
    def target_features(self) -> np.ndarray:
        if not hasattr(self, "_target_features"):
            raise AttributeError("Target features have not been set.")
        return self._target_features

    @property
    def descriptor_object(self):
        return self._descriptor_object

    @property
    def run_dir(self) -> str:
        """Directory where the data of the run is stored.

        The directory is named after the name of the run and is stored in the
        given save_dir.
        """
        return self._run_dir

    @property
    def structures_database(self) -> Database:
        """ASE Database to store selected structures of the run."""
        if not hasattr(self, "_structures_database"):
            self._structures_database = db.connect(
                os.path.join(self.run_dir, "structures.db")
            )
        return self._structures_database

    @property
    def global_statistics(self) -> tools.MultiStatistics | None:
        if not hasattr(self, "_global_statistics"):
            self._global_statistics = None
        return self._global_statistics

    def _get_global_logbook(self) -> tools.Logbook:
        """Generate the global logbook for the global_statistics."""
        global_log = tools.Logbook()
        global_stats_fields = []
        if self.global_statistics is not None:
            global_stats_fields = self.global_statistics.fields

        global_log.header = ["stage_id", "gen"] + global_stats_fields  # type: ignore
        return global_log

    @property
    def global_logbook(self) -> tools.Logbook:
        """A logbook for the global statistics.

        The logbook is used to store the global statistics for all
        generations of all stages.
        In addition to the global statistics, the logbook also stores
        the stage id and the generation number for each entry.
        If the global statistics are not set, the logbook only stores the
        stage id and the generation number.
        """
        if not hasattr(self, "_global_log"):
            self._global_log = self._get_global_logbook()

        return self._global_log

    def __update_stage_history(self, stage_id: int, relative_save_path: str):
        """Adds new entry to the :attr:`stage_history`."""
        self._stage_history["ID"].append(stage_id)
        self._stage_history["relative_save_path"].append(relative_save_path)

    def __create_global_statistics(
        self, global_stats_dict: dict[str, Callable[[Individual], float]] | None
    ) -> tools.MultiStatistics | None:
        """Method to create a MultiStatistics object from a dictionary of
        global statistics functions.

        The global statistics functions are used to calculate the mean, max,
        min and standard deviation for each iteration that modifies the
        population (e.g. in Genetic Algorithms they are calculated for each
        generation) of all stages of the optimization algorithm.

        :param global_stats_dict: Dictionary where the keys are the names of the
            statistics and the values are functions that calculate the
            statistics for an individual.
        """
        if global_stats_dict is None:
            return None

        stats_dict = {}
        for key, func in global_stats_dict.items():
            stats_dict[key] = tools.Statistics(key=func)

        mstats = tools.MultiStatistics(**stats_dict)
        mstats.register("avg", np.mean)
        mstats.register("max", np.max)
        mstats.register("min", np.min)
        mstats.register("std", np.std)

        return mstats

    def __create_run_dir(self, save_dir: str | os.PathLike) -> str:
        """Method to create a directory to store the data of the run.

        It is created as a subdir of the `save_dir` and named after the run
        name.

        :param save_dir: Directory where the run directory should be created.
        """
        run_dir = os.path.join(os.getcwd(), save_dir, self.name)

        # Check if dir already exists to not overwrite old data
        if os.path.isdir(run_dir):
            raise FileExistsError(
                f"Cannot create: {run_dir}!\n"
                f"There exists a run_dir for run '{self.name}' at {save_dir}.\n"
                "Either delete this directory, or change the name of the run!"
            )

        os.mkdir(run_dir)

        return run_dir

    def __get_time_string(self) -> str:
        """Get the current time as a string %Y_%m_%d_H%H_%M_%S."""
        now = datetime.datetime.now()
        date_string = now.strftime("%Y_%m_%d_H%H_%M_%S")
        return date_string

    def __save_stage_info(self, stage: Stage):
        """Method to save the info of a stage in a JSON file.

        The info of the stage is saved in a JSON file in the stage directory.
        The info is the :attr:`Stage.info_dict` of the stage with the added
        :data:`stage_id`, :attr:`Stage.type`, :attr:`Stage.name` and
        :attr:`Stage.description`.

        :param stage: Stage that should be saved.
        :param stage_dir: Directory where the info of the stage should be saved.
        """
        stage_info_dict = stage.info_dict.copy()
        stage_info_dict.update(
            {
                "id": stage.id,
                "type": stage.type(),
                "name": stage.name,
                "description": stage.description,
                "start_time": stage.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "start_time_ms": int(stage.start_time.timestamp() * 1000),
                "end_time": stage.end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time_ms": int(stage.end_time.timestamp() * 1000),
                "total_runtime": str(stage.end_time - stage.start_time),
                "total_runtime_ms": int(
                    (stage.end_time - stage.start_time).total_seconds() * 1000
                ),
            }
        )

        file_path = os.path.join(stage.stage_dir, "info.json")
        with open(file_path, "w") as f:
            json.dump(stage_info_dict, f)

        stage.logger.debug(f"Saved info.json of stage at {file_path}")

    def __set_up_stage(self, stage: Stage) -> str:
        """Method to set up the stage for a run.

        Adds the unique stage ID to the stage and creates a directory for the
        stage. Creates a new directory in the run directory with the name
        "stage_{:data:`stage_id`}".  Takes the :attr:`Stage.info_dict` of the
        stage and adds the :data:`stage_id`, :attr:`Stage.type`,
        :attr:`Stage.name` and :attr:`Stage.description` to the dictionary.
        Then saves the dictionary as a JSON file in the stage directory.

        :param stage: Stage that should be set up.
        """
        # Assign the stage ID to the stage
        stage.id = self.current_stage_id

        # Add the current time as the start time of the stage
        stage.set_start_time()

        # Create a directory for the stage in the run directory
        # stage should be saved in a directory relative to the run
        relative_stage_dir = f"stage_{stage.id}"
        stage_dir = os.path.join(self.run_dir, relative_stage_dir)
        os.mkdir(stage_dir)

        # Set the stage directory in the stage
        stage.stage_dir = stage_dir

        # Set up a logger for the stage
        stage_logger = setup_stage_logger(
            log_file_path=f"{stage_dir}/stage.log",
            run_name=self.name,
            stage_name=stage.name,
            log_level=self.log_level,
        )

        # Attach logger to the stage
        stage.logger = stage_logger

        stage_logger.info(f"Set up stage {stage.id}: {stage.name}")

        # Save the info of the stage in a JSON file in the stage directory
        self.__save_stage_info(stage)

        # Update the stage history with the currently run stage
        self.__update_stage_history(
            stage_id=stage.id, relative_save_path=relative_stage_dir
        )

        return stage_dir

    def store_file(
        self, original_file_path: str | os.PathLike, new_name: str | None = None
    ) -> Path:
        """Copy a file into ``self.run_dir``.

        :param original_file_path: Path to the existing file to copy.
        :param new_name: Optional name for the copied file inside ``self.run_dir``.
            If omitted, the original file name is preserved.

        :returns: The path of the copied file inside ``self.run_dir``.
        """
        assert os.path.isfile(original_file_path)

        src = Path(original_file_path)
        dst_name = new_name if new_name is not None else src.name
        dst = Path(self.run_dir) / dst_name

        shutil.copy2(src, dst)

        return dst

    def save_results(self):
        """Method to save the results of the run in a JSON file.

        The results of the run are saved in a JSON file in the run directory.
        The results are the global statistics of the run if they are set.  The
        keys of the dictionary are "names", "functions" and "results". The
        values are lists where each index corresponds to one statistic. The
        results are stored in a dictionary with the keys "stage_id", "gen",
        "min", "max", "avg" and "std" for each statistic.
        """
        # Create a dictionary to store the global statistics if they are set
        global_stats_dict = {}
        if self._global_statistics_dict is not None:
            global_stats_dict["names"] = [
                name for name in self._global_statistics_dict.keys()
            ]
            global_stats_dict["functions"] = [
                func.__name__ for func in self._global_statistics_dict.values()
            ]

            # Loop over the chapters of the logbook and save the statistics
            global_stats_dict["results"] = []
            for name in self.global_logbook.chapters.keys():
                stat_log = self.global_logbook.chapters[name]
                stat_result = {
                    "gen": stat_log.select("gen"),
                    "min": stat_log.select("min"),
                    "max": stat_log.select("max"),
                    "avg": stat_log.select("avg"),
                    "std": stat_log.select("std"),
                    "stage_id": stat_log.select("stage_id"),
                }
                global_stats_dict["results"].append(stat_result)
        elif len(self.global_logbook) > 0:
            # For the chase that global stats are not set but there are records in the log
            # warn user that nothing will be recorded
            warnings.warn(
                "Global statistics were never set, so no results will be recorded."
            )

        # Save the global statistics in a JSON file
        file_path = os.path.join(self.run_dir, "global_statistics.json")
        with open(file_path, "w") as f:
            json.dump(global_stats_dict, f)

    def save_info(self):
        """Stores info of the run to json file.

        File is located at info.json in :attr:`run_dir`.  Stores :attr:`name`,
        :attr:`description`, :attr:`start_time`, :attr:`end_time`,
        `total_runtime`, :attr:`_stage_history`. The stage history is a dict of
        stage ids and the corresponding directory paths (relative to
        :attr:`run_dir`), where the data of the stage is stored.

        All times are stored as timestemp as well as millisecs since epoch.
        """
        file_path = os.path.join(self.run_dir, "info.json")
        info_dict = {
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_time_ms": int(self.start_time.timestamp() * 1000),
            "end_time": self.end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time_ms": int(self.end_time.timestamp() * 1000),
            "total_runtime": str(self.end_time - self.start_time),
            "total_runtime_ms": int(
                (self.end_time - self.start_time).total_seconds() * 1000
            ),
            "stage_history": self._stage_history,
        }
        with open(file_path, "w") as f:
            json.dump(info_dict, f)

    def run(self, population: Population, stage: Stage) -> Population:
        """Method to run a stage of the optimization algorithm.

        This method runs a stage of the optimization algorithm and makes the
        stage save its results.
        Manages the stage ID and directory where the results are saved.
        Also saves the info.json of the run and the global_statisitics.json.
        If already present overwrites them.

        :param population: Population that should be optimized.
        :param stage: Stage that should be run.
        """
        if not self.global_statistics_dict:
            warnings.warn(
                "No global_statistics_dict set, so no global stats will be tracked.",
                UserWarning,
            )

        # Update the current stage ID, to ensure the stages have unique IDs
        self.current_stage_id += 1

        # Set the start time of the run to the current time
        if self.current_stage_id == 1:
            self._start_time = datetime.datetime.now()

        # Create a directory for the stage first, to ensure data can be saved
        stage_dir = self.__set_up_stage(stage)

        # Run the stage and save the results
        self.logger.info(
            f"Run {self.name}: Running stage: {stage.name}, ID: {stage.id}"
        )
        self.logger.debug(f"Population size: {population.size}")
        population = stage.run(
            population=population,
            global_log=self.global_logbook,
            global_stats=self.global_statistics,
        )

        self.logger.info(
            f"Run {self.name}: Finished stage: {stage.name}, ID: {stage.id}"
        )
        self.logger.debug(f"Saving at directory: {stage_dir}")
        stage.save_results(
            save_dir=stage_dir,
            structures_db=self.structures_database,
            global_statistics_dict=self._global_statistics_dict,
        )

        # Set the end time of the stage to the current time
        stage.set_end_time()

        # Save the stage_info_dict again, to update data. E.g. number of generations
        self.__save_stage_info(stage)

        # Set the end time of the run. Will be overwritten if run again.
        self.set_end_time()

        # Overwrite the info.json in the run directory
        self.save_info()

        # Overwrite the global_statistics.json in the run directory
        self.save_results()

        return population
