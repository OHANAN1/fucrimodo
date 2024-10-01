from typing import Callable
from ase import db
from ase.db.core import Database
from deap import tools
import numpy as np

from fucrimodo.core.modules.individual import Individual
from .modules import Stage, Population
import os
import pickle
import datetime

class MultiStageSearch:
    """Class to run the multi-stage optimization algorithm.

    The multi-stage optimization algorithm is used to run stages
    of optimization algorithms.

    :param save_dir: Directory where a dictionary should be created to store
        the data of the run.
    :param descriptive_name: Optional name of the run. If no name is given,
        the current time and date is used. Saved to :attr:`name`.
    :param global_statistics_dict: Optional dictionary, where the keys are the
        names of the statistics and the values are functions that calculate the
        statistics for an individual. The statistics are calculated for each
        iteration that modifies the population (e.g. in Genetic Algorithms they
        are calculated for each generation) of all stages of the optimization
        algorithm.
    """
    def __init__(
        self,
        save_dir: str,
        descriptive_name: str|None = None,
        global_statistics_dict: dict[str, Callable[[Individual], float]] | None = None,
    ) -> None:
        # If no descriptive name is given, use the current time and date.
        # Define name attribute without setter, since it should never be changed
        if descriptive_name is None:
            self._name = self.__get_time_string()
        else:
            self._name = descriptive_name

        # Create the dictionary to store the data of the run
        self._run_dir = self.__create_run_dir(save_dir)

        # Create the global statistics and the logbook
        self._global_statistics = self.__create_global_statistics(
            global_statistics_dict
        )
        self._global_log = self.global_logbook

        # Set the current stage id to 0
        self.current_stage_id = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def run_dir(self) -> str:
        """Directory where the data of the run is stored.
        The directory is named after the name of the run or the time of the
        initialization of the class and is stored in the given save_dir.
        """
        return self._run_dir

    @property
    def crystal_database(self) -> Database:
        """ASE Database to store selected crystal structures of the run."""
        if not hasattr(self, "_crystal_database"):
            self._crystal_database = db.connect(f"{self.run_dir}/crystals.db")
        return self._crystal_database

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

            self._global_log.header = ['stage_id', 'gen'] + global_stats_fields # type: ignore

        return self._global_log

    def save_results(self):
        file_path = os.path.join(self.run_dir, "global_logbook.pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.global_logbook, f)

    def run(self, population: Population, stage: Stage) -> Population:
        """Method to run a stage of the optimization algorithm.

        This method runs a stage of the optimization algorithm and makes the
        stage save the results of the optimization algorithm.
        Manages the stage ID and directory where the results are saved.

        :param population: Population that should be optimized.
        :param stage: Stage that should be run.
        """
        # Update the current stage ID and assign it to the stage to make it 
        # identifiable in the results
        self.current_stage_id += 1
        stage.id = self.current_stage_id

        # Create a directory for the stage first, to ensure data can be saved
        save_dir = self.__set_up_stage_dir(self.current_stage_id)

        # Run the stage and save the results
        print(f"Running stage {self.current_stage_id}:")
        print(f"Stage ID: {stage.id}")
        print(f"Poulation size: {population.size}")
        population = stage.run(
            population=population,
            global_log=self.global_logbook,
            global_stats=self.global_statistics, 
        )

        print(f"Saving results of stage {self.current_stage_id}: {stage.name}")
        stage.save_results(
            save_dir = save_dir,
            crystals_db = self.crystal_database
        )
        return population

    def __create_global_statistics(
        self, 
        global_stats_dict: dict[str, Callable[[Individual], float]] | None
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
        run_dir = os.path.join(save_dir, self.name)
        os.mkdir(run_dir)

        return run_dir

    def __get_time_string(self) -> str:
        now = datetime.datetime.now()
        date_string = now.strftime("%Y_%m_%d_H%H_%M_%S")
        return date_string

    def __set_up_stage_dir(self, stage_id: int) -> str:
        """Method to create a directory for the stage with the given ID.

        In the run directory a new directory is created with the name of the
        stage ID.

        :param stage_id: ID of the stage for which the directory should be
            created.
        """
        stage_dir = os.path.join(self.run_dir, f"stage_{stage_id}")
        os.mkdir(stage_dir)

        return stage_dir
