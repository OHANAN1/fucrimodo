import os
import random
import numpy as np
from time import sleep
from ase.db.core import Database
import ase.db
from typing import Callable
from deap import tools
from multiprocessing import Pool

from fucrimodo.core.utils.log_utils import setup_stage_logger
from fucrimodo.core.modules import Stage, Individual, Population, PopulationSelection
from fucrimodo.customs.ga_stage import GAStage
from fucrimodo.customs.population_generator import convert_ase_atoms_to_individual
from fucrimodo.customs.population_selections import SelectAllPopulation


# Method that is run in parallel to perform the stages
# It needs to be defined outside of the class to be pickable
def perform_stage(
    stage: GAStage,
    population: Population,
    global_log: tools.Logbook,
    global_stats: tools.MultiStatistics,
    crystal_db: Database,
    seed: int,
) -> tuple[Population, tools.Logbook]:
    """Method to perform a stage in parallel.

    Is defined outside of the class to be pickable. This also means that all
    the parameters and return values also need to be pickable.
    Do not use lambda functions or other non-pickable objects in this method.
    Use different seeds for each stage to make sure they are reproducible and
    do not conflict with each other.
    """
    random.seed(seed)
    np.random.seed(seed)

    population = stage.run(population, global_log, global_stats)

    stage.save_results(stage.stage_dir, crystal_db)

    return population, global_log


class GAParallelStage(Stage):
    """Class to run multiple GA stages in parallel.

    The stages are run in parallel using the multiprocessing module. The stages
    need to be instances of the GAStage class and all parameters of the stages
    need to be pickable. This means no lambda functions or other non-pickable
    objects can be used in the stages.

    :param name: Name of the parallel stage.
    :param description: Description of the parallel stage.
    :param stage_list: List of GAStage instances to run in parallel.
    :param survivor_selection: Survivor selection method to use after the
        stages are run. If None, no survivor selection is performed.
        If set, the number of individuals in the population will be kept
        constant.
    :param n_processes: Number of processes to use to run the stages in
        parallel.
    :param random_seed: Random seed to use for the parallel stage.
        Should be set to the global seed of the algorithm. Each stage is then
        run with a unique seed based on this seed.
    :param verbose: Toggle to print information to the console.
    """
    def __init__(
        self,
        name: str,
        description: str,
        stage_list: list[GAStage],
        survivor_selection: PopulationSelection | None = None,
        n_processes: int = 4,
        random_seed: int = 42,
        verbose: bool = True
    ):
        super().__init__(name, description)
        self.stage_list = stage_list
        self.n_processes = n_processes
        self.random_seed = random_seed
        self.verbose = verbose
        self.survivor_selection = survivor_selection

    @property
    def info_dict(self) -> dict:
        info_dict = {
            "type": "GAParallelStage",
            "name": self._name,
            "description": self._description,
            "stage_list": [stage.info_dict for stage in self.stage_list]
        }
        return info_dict

    def __write_local_crystals_db_to_global_crystals_db(
        self,
        global_crystals_db: Database,
        global_stats_dict: dict[str, Callable[[Individual], float]] | None = None,
    ) -> None:
        """Method to write the temporary crystals database, that was used to
        store the results of the stages, to the global crystals database.

        Updates the stage IDs of the local crystals database to include the
        stage ID of the parallel stage and calculates the global statistics for
        each individual.
        """

        for row in self.local_crystals_db.select():
            local_stage_id = row["stage_id"]
            # Adjust the stage ID to include the stage ID of the parallel stage
            # Use ',' as separator to not make it a float
            row["stage_id"] = f"{self.id},{local_stage_id}"
            atoms = row.toatoms()
            key_value_pairs = row.key_value_pairs
            
            # Calculate global statistics for the individual if they exist
            if global_stats_dict is not None:
                ind = convert_ase_atoms_to_individual(atoms)
                if global_stats_dict is not None:
                    for key, func in global_stats_dict.items():
                        key_value_pairs[key] = func(ind)

            global_crystals_db.write(atoms, key_value_pairs)

    def save_results(
        self,
        save_dir: str,
        crystals_db: Database,
        global_statistics_dict: dict[str, Callable[[Individual], float]] | None = None
    ) -> None:
        self.logger.info("Saving results of parallel stage")

        self.__write_local_crystals_db_to_global_crystals_db(
            crystals_db,
            global_statistics_dict
        )

        # Remove the temporary crystals database
        self.logger.debug(
            f"Removing temporary crystals database at: {self.local_crystals_db_path}"
        )
        os.remove(self.local_crystals_db_path)

    def __set_up_stage(self, stage: GAStage, stage_id: int) -> str:
        """Method to set up each of the stages in the parallel run."""
        # Assign the stage ID to the stage
        stage.id = stage_id

        # Create a directory for the stage in the stage directory of the 
        # parallel stage
        relative_stage_dir = f"stage_{stage_id}"
        stage_dir = os.path.join(self.stage_dir, relative_stage_dir)
        os.mkdir(stage_dir)

        # Attach the stage directory to the stage
        stage.stage_dir = stage_dir

        # Set up a logger for each stage
        stage_logger, _ = setup_stage_logger(
            log_file_path=f"{stage_dir}/stage.log",
            run_name=self.name,
            stage_name=stage.name,
            log_level=self.logger.level
        )

        # Attach logger to the stage
        stage.logger = stage_logger

        stage_logger.info(f"Set up stage {stage_id}: {stage.name}")

        # Turn off verbosity of stage to not print to console
        stage.verbose = False

        return stage_dir

    def __set_up_self(self) -> None:
        """Method to set up the parallel stage."""

        # Set up each stage
        for stage_index, stage in enumerate(self.stage_list):
            self.__set_up_stage(stage, stage_index + 1)

        self.logger.info("Set up parallel stage")

        # Create a crystal database where the stages can write their results to.
        # The structures will be added to the global crystals db when the 
        # save_results method is called. The db is then removed.
        self.local_crystals_db_path = os.path.join(self.stage_dir, "crystals.db")
        self.local_crystals_db = ase.db.connect(self.local_crystals_db_path)

    def __write_logs_to_global_log(
        self,
        global_log: tools.Logbook,
        stage_logs: list[tools.Logbook]
    ) -> int:
        # Get the maximum number of generations that have been run in any of
        # the stages
        max_gen = max([len(log.select("gen")) for log in stage_logs])

        # Get the last generation that has been run in the global log
        # to add it to each new generation
        try: 
            last_gen_global_log = global_log.select("gen")[-1]
        except:
            last_gen_global_log = 0

        if last_gen_global_log is None:
            last_gen_global_log = 0
        if not isinstance(last_gen_global_log, int):
            last_gen_global_log = 0

        # Preset the global gen with the last generation that has been run
        global_gen = last_gen_global_log

        # loop over the maximum number of generations that was performed in any
        # of the stages
        for gen in range(1, max_gen+1):
            # Add the global generation to the stage generation
            global_gen = last_gen_global_log + gen

            # Create a record for the generation with the chapters in one of
            # the stage logs
            # Use very highest and lowest values for min and max so they are
            # always replaced
            record_for_gen = {
                chapter: {
                    "min": np.inf,
                    "max": -np.inf,
                    "mean": 0,
                    "std": 0
                } for chapter in stage_logs[0].chapters.keys() # Use the chapters of the first stage log, dont use chapters of the global log since it is potentially empty
            }

            # Loop over all the stage logs
            for log in stage_logs:

                # If the current stage log does not have the generation
                # continue to the next stage log
                if len(log.select("gen")) - 1 < gen - 1:
                    continue
                else:
                    # Loop over all the chapters in the stage log
                    for chapter in log.chapters.keys():
                        # Get the min value for the chapter in the generation
                        min_val = log.chapters[chapter].select("min")[gen-1]
                        if isinstance(min_val, list):
                            min_val = min(min_val)

                        # If the min value is lower than the current min value
                        # replace it
                        if record_for_gen[chapter]["min"] > min_val:
                            record_for_gen[chapter]["min"] = min_val

                        # Get the max value for the chapter in the generation
                        max_val = log.chapters[chapter].select("max")[gen-1]
                        if isinstance(max_val, list):
                            max_val = max(max_val)

                        # If the max value is higher than the current max value
                        # replace it
                        if record_for_gen[chapter]["max"] < max_val:
                            record_for_gen[chapter]["max"] = max_val

            # Add the record for the generation to the global log
            global_log.record(
                gen=global_gen, stage_id=self.id, **record_for_gen
            )
            if self.verbose:
                print(global_log.stream)

        return last_gen_global_log + max_gen

    def __create_empty_copy_of_global_log(
        self,
        global_log: tools.Logbook,
        global_statistics: tools.MultiStatistics | None = None
    ) -> tools.Logbook:
        """Method to create an copy of the global log that is empty."""
        # NOTE: The notebook needs to be created from scratch since copying it
        #   leads to a bug when the algorithm is run in parallel where it is
        #   not correctly updated.
        #   Therefore, the global log is created from scratch for each stage.
        global_log_copy = tools.Logbook()

        global_stats_fields = []
        if global_statistics is not None:
            global_stats_fields = global_statistics.fields

        global_log_copy.header = ['stage_id', 'gen'] + global_stats_fields # type: ignore

        return global_log_copy

    def run(
        self,
        population: Population,
        global_log: tools.Logbook,
        global_stats: tools.MultiStatistics | None
    ) -> Population:
        self.__set_up_self()

        if self.verbose:
            print(f"Running {self.name} with {self.n_processes} processes...")
        with Pool(self.n_processes) as p:
            # Run the stages in parallel with a pool of processes
            results = p.starmap_async(
                perform_stage, [
                    (
                        stage,
                        population,
                        self.__create_empty_copy_of_global_log(global_log, global_stats),
                        global_stats,
                        self.local_crystals_db,
                        # Set unique seed for each stage so it is reproducible 
                        # and seeds do not conflict, should be unique even 
                        # from main seed so it is not reset (therefore I use +1)
                        self.random_seed + i + 1
                    )
                    for i, stage in enumerate(self.stage_list)
                ]
            )

            # Wait for the results to be ready with fun little animation
            wait_indicator = ["/", "-", "\\", "|"]
            while not results.ready():
                if self.verbose:
                    if len(wait_indicator) == 0:
                        wait_indicator = ["/", "-", "\\", "|"]
                    print(f"Waiting for results to be ready... {wait_indicator.pop()}", end="\r")

                    sleep(0.5)

            # Unpack the results and combine the populations
            # Stages also need to be unpacked to use their attributes
            individuals = []
            stage_global_logs = []
            for pop, glob_log in results.get():
                individuals += pop.individuals
                stage_global_logs.append(glob_log)

        # Add the global logs from the stages to the global log
        global_gen = self.__write_logs_to_global_log(
            global_log,
            stage_global_logs
        )

        # If set, perform survivor selection
        if self.survivor_selection is not None:
            individuals = self.survivor_selection.select(
                individuals, population.size
            )

        # Update the population with the individuals from the stages
        population.individuals = individuals

        # Update the population to have the correct generation number
        population.generation = global_gen

        if self.verbose:
            print(f"Finished running {self.name} with {self.n_processes} processes.")

        return population
