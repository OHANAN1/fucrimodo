import datetime
import json
import logging
import os
from typing import Callable

import numpy as np
from ase import db
from ase.db.core import Database
from deap import tools

from fucrimodo.core.modules.individual import Individual
from fucrimodo.core.utils.log_utils import setup_run_logger, setup_stage_logger

from .modules import Population, Stage


class MultiStageSearch:
    """Class to run the multi-stage optimization algorithm.

    The multi-stage optimization algorithm is used to run stages
    of optimization algorithms.

    :param save_dir: Directory where a dictionary should be created to store
        the data of the run.
    :param target_features: Array with the target features that the
        optimization algorithm should invert.
    :param descriptor_object: Object of the descriptor that is used to
        calculate the features of the individuals.
    :param descriptive_name: Optional name of the run. If no name is given,
        the current time and date is used. Saved to :attr:`name`.
    :param global_statistics_dict: Dictionary, where the keys are the
        names of the statistics and the values are functions that calculate the
        statistics for an individual. The statistics are calculated for each
        iteration that modifies the population (e.g. in Genetic Algorithms they
        are calculated for each generation) of all stages of the optimization
        algorithm. If not initialized at start must be set later.
    :param log_level: Log level of the global logger. Set to logging.INFO to
        see the progress of the run.
    :param verbose: If set to True, the global logger also logs to the console
        in addition to the log file.
    """

    def __init__(
        self,
        save_dir: str,
        target_features: np.ndarray,
        descriptor_object,
        descriptive_name: str | None = None,
        description: str = "",
        global_statistics_dict: dict[str, Callable[[Individual], float]] | None = None,
        log_level: int = logging.INFO,
        verbose: bool = True,
    ) -> None:
        # If no descriptive name is given, use the current time and date.
        # Define name attribute without setter, since it should never be changed
        if descriptive_name is None:
            self._name = self.__get_time_string()
        else:
            self._name = descriptive_name

        self._description = description

        # Set the target features of the optimization algorithm
        self._target_features = target_features

        # Set the descriptor object that is used to calculate the features
        self._descriptor_object = descriptor_object

        # Create the dictionary to store the data of the run
        self._run_dir = self.__create_run_dir(save_dir)

        # Save the global statistics dictionary to access it during saving
        self.global_statistics_dict = global_statistics_dict

        # Set the current stage id to 0
        self.current_stage_id = 0

        # Set up the global logger for the run
        self.logger = setup_run_logger(
            log_file_path=f"{self.run_dir}/run.log",
            run_name=self.name,
            log_level=log_level,
            verbose=verbose,
        )

        self.logger.info(f"Initialized run {self.name}")

    @property
    def name(self) -> str:
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
        # set new log level attribute
        self._log_level = value

        # update the log level of the global logger
        self.logger.setLevel(value)

    @property
    def start_time(self) -> datetime.datetime:
        return self._start_time

    @property
    def end_time(self) -> datetime.datetime:
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

        Only for information. The number of parallel jobs must be set in the
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
        return self._global_statistics_dict

    @global_statistics_dict.setter
    def global_statistics_dict(
        self, value: dict[str, Callable[[Individual], float]] | None
    ):
        """Setter for the global statistics dictionary.

        The setter also creates the global statistics and the logbook for the
        new statistics.
        """
        self._global_statistics_dict = value
        # Create the global statistics and the logbook for the new statistics
        if value is None:
            self._global_statistics = None
        else:
            self._global_statistics = self.__create_global_statistics(
                self._global_statistics_dict
            )
            self._global_log = self.global_logbook

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
        The directory is named after the name of the run or the time of the
        initialization of the class and is stored in the given save_dir.
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
        return self._global_statistics

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
            self._global_log = tools.Logbook()

            global_stats_fields = []
            if self.global_statistics is not None:
                global_stats_fields = self.global_statistics.fields

            self._global_log.header = ["stage_id", "gen"] + global_stats_fields  # type: ignore

        return self._global_log

    @property
    def stage_history(self) -> dict[str, list]:
        """Dictionary to store the history of the stages.

        The dictionary stores ordered lists of the stage IDs and
        paths to the directories of the stages relative to the directory the
        run was saved in. Each index in the lists corresponds to one stage.

        :returns: History dict of the stages. Keys are: "ID", "relative_save_path"
        """
        if not hasattr(self, "_stage_history"):
            self._stage_history = {
                "ID": [],
                "relative_save_path": [],
            }
        return self._stage_history

    def __update_stage_history(self, stage_id: int, relative_save_path: str):
        """Adds new entry to the :attr:`stage_history`."""
        self.stage_history["ID"].append(stage_id)
        self.stage_history["relative_save_path"].append(relative_save_path)

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

    def __create_run_dir(self, save_dir: str) -> str:
        """Method to create a directory to store the data of the run.

        In the given save_dir a new directory is created with the name of the
        run.

        :param save_dir: Directory where the run directory should be created.
        """
        run_dir = os.path.join(os.getcwd(), save_dir, self.name)

        # Check if dir already exists to not overwrite old data
        # TODO: Replace this with the option to restart run from old data
        if os.path.isdir(run_dir):
            raise FileExistsError(
                f"Cannot create: {run_dir}!\n"
                f"There exists a run_dir for run '{self.name}' at {save_dir}.\n"
                "Either delete this directory, or change the name of the run!"
            )

        os.mkdir(run_dir)

        return run_dir

    def __get_time_string(self) -> str:
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

        Adds the stage ID to the stage and creates a directory for the stage.
        For this a new directory is created in the run directory with the name
        "stage_{:data:`stage_id`}".
        Takes the :attr:`Stage.info_dict` of the stage and adds the
        :data:`stage_id`, :attr:`Stage.type`, :attr:`Stage.name` and
        :attr:`Stage.description` to the dictionary.
        Then saves the dictionary as a JSON file in the stage directory.

        :param stage: Stage that should be set up.
        :param stage_id: A unique ID that should be assigned to the stage.
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
        stage_logger, log_name = setup_stage_logger(
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

    def save_results(self):
        """Method to save the results of the run in a JSON file.

        The results of the run are saved in a JSON file in the run directory.
        The results are the global statistics of the run if they are set.
        The keys of the dictionary are "names", "functions" and "results".
        The values are lists where each index corresponds to one statistic.
        The results are stored in a dictionary with the keys "stage_id", "gen",
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

        # Save the global statistics in a JSON file
        file_path = os.path.join(self.run_dir, "global_statistics.json")
        with open(file_path, "w") as f:
            json.dump(global_stats_dict, f, indent=4)

    def save_info(self):
        file_path = os.path.join(self.run_dir, "info.json")
        info_dict = {
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": self.end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_runtime": str(self.end_time - self.start_time),
            "stage_history": self.stage_history,
        }
        with open(file_path, "w") as f:
            json.dump(info_dict, f, indent=4)

    def run(self, population: Population, stage: Stage) -> Population:
        """Method to run a stage of the optimization algorithm.

        This method runs a stage of the optimization algorithm and makes the
        stage save the results of the optimization algorithm.
        Manages the stage ID and directory where the results are saved.
        Also saves the info.json of the run and the global_statisitics.json.
        If already present overwrites them.

        :param population: Population that should be optimized.
        :param stage: Stage that should be run.
        """

        # TODO: Make global statistics optional
        if not self.global_statistics_dict:
            raise ValueError(
                "Please set a global_statistics_dict, so stats on the run can be tracked."
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
