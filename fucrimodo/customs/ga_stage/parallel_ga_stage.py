import os
import random
import numpy as np
from time import sleep
from ase.db.core import Database
import ase.db
from typing import Callable
from deap import tools
from multiprocessing import Pool
import json

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

    # Set the start time of the stage
    stage.set_start_time()

    population = stage.run(population, global_log, global_stats)

    # Set the end time of the stage to the current time
    stage.set_end_time()

    # Save stage results
    stage.save_results(stage.stage_dir, crystal_db)

    # Save stage info
    stage_info_dict = stage.info_dict.copy()
    stage_info_dict.update({
        "id": stage.id,
        "type": stage.type(),
        "name": stage.name,
        "description": stage.description,
        "start_time": stage.start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": stage.end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime": str(stage.end_time - stage.start_time),
    })

    file_path = os.path.join(stage.stage_dir, "info.json")
    with open(file_path, "w") as f:
        json.dump(stage_info_dict, f, indent=4)

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
    :param parent_selection: Parent selection method to use on the population
        before the stages are run. If None, all individuals in the population
        are used as the population.
    :param parent_ratio: Ratio of the population that is selected by the
        parent selection method.
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
        parent_selection: PopulationSelection | None = None,
        parent_ratio: float = 1.0,
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
        self.parent_selection = parent_selection
        self.parent_ratio = parent_ratio

    def __get_stage_history(self) -> dict:
        """Method to get the stage history of the parallel stage.

        The stage history is a dictionary with the stage IDs, names and info
        dicts of the stages in the parallel stage.
        Uses the info dicts that were saved in the info.json files of the stages
        at the end of each run if they exist.
        """
        # Load a stage history for each of the parallel stage
        stage_history = {
            "names": [stage.name for stage in self.stage_list],
            "info_dicts": [stage.info_dict for stage in self.stage_list]
        }

        # Try to load the final info dicts that were saved in the info.json
        # files of each stage at the end of each run
        for i, stage in enumerate(self.stage_list):

            # Check if the stage directory can be loaded.
            # if not, continue to the next stage
            try:
                stage_dir = stage.stage_dir
            except:
                continue

            # If the stage directory is None, it is not set jet
            # continue to the next stage
            if stage_dir is None:
                continue

            # Check if the info file exists
            info_file_path = os.path.join(stage_dir, "info.json")
            if os.path.isfile(info_file_path):

                # Load the info dict from the file
                with open(info_file_path, "r") as f:
                    stage_info_dict = json.load(f)

                    # overwrite the info dict in the stage history
                    # with the loaded info dict
                    stage_history["info_dicts"][i] = stage_info_dict

        return stage_history

    @property
    def info_dict(self) -> dict:
        stage_history = self.__get_stage_history()
        n_generations = np.sum([
            stage_info_dict["n_generations"] 
            for stage_info_dict in stage_history["info_dicts"]
        ]).tolist()

        info_dict = {
            "type": "GAParallelStage",
            "name": self._name,
            "description": self._description,
            "n_processes": self.n_processes,
            "n_generations": n_generations,
            "stage_history": self.__get_stage_history(),
            "survivor_selection": self.survivor_selection.__repr__(),
            "parent_selection": self.parent_selection.__repr__(),
            "parent_ratio": self.parent_ratio,
        }

        # Add these entries, so the info_dict is consistent with the other
        # stages
        info_dict["break_condition"] = None
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

    def __combine_stage_data(self) -> None:
        """Combines the data of all the stages in the parallel stage into shared files

        Each GAStage stores its data in a separate directory. The files are
        called: 'mutations.json', 'crossovers.json', 'fitnesses.json',
        This method combines the data of all the stages into shared files in the
        parallel stage directory.
        """
        # Define the dictionaries to store the combined data
        # Must have the same keys as the data of the stages
        # Look at the GAStage methods to see the keys used
        mutation_dict_combined = {
            "names": [],
            "weights": [],
            "reprs": [],
            "hashes": [],
            "results": [],
            "stage_id": []
        }
        crossover_dict_combined = {
            "names": [],
            "weights": [],
            "reprs": [],
            "hashes": [],
            "results": [],
            "stage_id": []
        }
        fitnesses_dict_combined  = {
            "names": [],
            "weights": [],
            "reprs": [],
            "titles": [],
            "hashes": [],
            "results": []
        }

        # Define a method to append the data of a stage to the combined
        # dictionaries
        def append_json_to_combined_dict(
            json_file_path: str, combined_dict: dict, stage_id: int
        ) -> None:
            # Load the data of the stage from the json file
            with open(json_file_path, "r") as f:
                data = json.load(f)

            # Append the stage_id to idetify where the data comes from
            data["stage_id"] = [stage_id] * len(data["names"])

            # Append the data of the stage to the combined dictionaries
            # Must have the same keys
            for key in combined_dict.keys():
                combined_dict[key] += data[key]

        for stage in self.stage_list:
            parallel_stage_dir = stage.stage_dir

            # Append the data of the stage to the combined dictionaries
            append_json_to_combined_dict(
                os.path.join(parallel_stage_dir, "mutations.json"),
                mutation_dict_combined,
                stage.id
            )
            append_json_to_combined_dict(
                os.path.join(parallel_stage_dir, "crossovers.json"),
                crossover_dict_combined,
                stage.id
            )
            append_json_to_combined_dict(
                os.path.join(parallel_stage_dir, "fitnesses.json"),
                fitnesses_dict_combined,
                stage.id
            )

        # Write the combined data to the parallel stage directory
        with open(os.path.join(self.stage_dir, "mutations.json"), "w") as f:
            json.dump(mutation_dict_combined, f, indent=4)
        with open(os.path.join(self.stage_dir, "crossovers.json"), "w") as f:
            json.dump(crossover_dict_combined, f, indent=4)
        with open(os.path.join(self.stage_dir, "fitnesses.json"), "w") as f:
            json.dump(fitnesses_dict_combined, f, indent=4)

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

        self.__combine_stage_data()

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

        if self.parent_selection is not None:
            n_parents = int(self.parent_ratio * population.size)
            population.individuals = self.parent_selection.select(
                population.individuals, n_parents
            )

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
