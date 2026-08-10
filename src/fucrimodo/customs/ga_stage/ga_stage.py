import json
import os
from typing import Any, Callable, Sequence
import numpy as np

from ase.db.core import Database
from deap import tools
from ...core.abstracts import (
    FitnessFunction,
    PopulationSelection,
    Stage,
)
from ...core import Individual, Population

from ..break_conditions import BreakCondition
from .crossovers import Crossover
from .genetic_algorithm import GeneticAlgorithm
from .mutations import Mutation


class GAStage(Stage):
    def __init__(
        self,
        name: str,
        fitness_functions: Sequence[FitnessFunction | tuple[FitnessFunction, float]],
        crossover_list: Sequence[Crossover | tuple[Crossover, float]],
        mutation_list: Sequence[Mutation | tuple[Mutation, float]],
        mutation_probability: float,
        crossover_probability: float,
        break_condition: BreakCondition,
        parent_selection: PopulationSelection,
        survivor_selection: PopulationSelection,
        parent_ratio: float = 0.5,
        description: str = "",
        save_n_structures: int = 10,
        rng: None | np.random.Generator = None,
    ):
        super().__init__(name, description)

        # Store initial config, so that copy of class can constructed
        self._cfg = dict(
            name=name,
            fitness_functions=fitness_functions,
            crossover_list=crossover_list,
            mutation_list=mutation_list,
            mutation_probability=mutation_probability,
            crossover_probability=crossover_probability,
            break_condition=break_condition,
            parent_selection=parent_selection,
            survivor_selection=survivor_selection,
            parent_ratio=parent_ratio,
            description=description,
            save_n_structures=save_n_structures,
        )

        fitness_funcs, fitness_weights = self._seperate_object_weight_tuples(
            fitness_functions
        )
        cross_list, crossover_weights = self._seperate_object_weight_tuples(
            crossover_list
        )
        mut_list, mutation_weights = self._seperate_object_weight_tuples(mutation_list)

        self.ga_runner = GeneticAlgorithm(
            fitness_functions=fitness_funcs,
            fitness_weights=fitness_weights,
            crossover_list=cross_list,
            crossover_weights=crossover_weights,
            mutation_list=mut_list,
            mutation_weights=mutation_weights,
            mutation_probability=mutation_probability,
            crossover_probability=crossover_probability,
            break_condition=break_condition,
            survivor_selection=survivor_selection,
            parent_selection=parent_selection,
            parent_ratio=parent_ratio,
            save_n_best_individuals=save_n_structures,
            rng=rng,
        )

    @property
    def stop_event(self) -> Any | None:
        """Optional attribute to stop the GA if `stop_event.is_set()` is True.

        Use a `multiprocessing.Event` to set the stop event.
        """
        if not hasattr(self, "_stop_event"):
            return None

        return self._stop_event

    @stop_event.setter
    def stop_event(self, value) -> None:
        self._stop_event = value

    def _seperate_object_weight_tuples(
        self, value: Sequence[Any | tuple[object, float]]
    ) -> tuple[list, tuple]:
        """
        Seperates the objects and the weights in the tuple list.
        If the sequence entry is not a tuple, the weight is set to 1.

        :return: tuple of the objects as a list and the weights as a tuple.
        """
        objects = []
        weights = ()
        for val in value:
            if isinstance(val, tuple):
                objects.append(val[0])
                weights += (val[1],)
            else:
                objects.append(val)
                weights += (1.0,)

        return objects, weights

    def _save_hall_of_fame(
        self,
        database: Database,
        hall_of_fame: tools.HallOfFame,
        fitness_functions: Sequence[FitnessFunction],
        global_stats_dict: dict[str, Callable] | None = None,
    ) -> None:
        """Saves the hall of fame to the database and adds the fitness to
        each individual.
        """
        for ind in hall_of_fame:
            key_value_pairs = {"stage_id": self.id}

            # Add the fitness values to the key value pairs
            for i in range(len(fitness_functions)):
                fitness_name = fitness_functions[i].db_title
                key_value_pairs[fitness_name] = ind.fitness.values[i]

            # Calculate global statistics for each individual if set
            if global_stats_dict is not None:
                for key, func in global_stats_dict.items():
                    key_value_pairs[key] = func(ind)

            database.write(ind, key_value_pairs)

    def _save_crossovers(self, save_dir: str):
        """Saves the crossover logbook and information to a json file.

        The json file will contain the following information:

        - 'names': The names of the crossovers
        - 'weights': The weights of the crossovers
        - 'reprs': The representations of the crossovers
        - 'hashes': The hashes of the crossovers
        - 'results': A list of dictionaries containing the following information:

            - 'gen': The generation number
            - 'called': The number of times the crossover was called
            - 'failed': The number of times the crossover failed
            - 'survivor': The number of times the offspring was selected as a survivor

        :param save_dir: The directory to save the file to.
        """
        # Set up a dictionary to store crossover information and results
        crossover_dict = {
            "names": [
                cross.__class__.__name__ for cross in self.ga_runner.crossover_list
            ],
            "weights": list(self.ga_runner.norm_crossover_weights),
            "reprs": [cross.__repr__() for cross in self.ga_runner.crossover_list],
            "hashes": [cross.__hash__() for cross in self.ga_runner.crossover_list],
            "results": [],
        }
        # Loop over all chapters in the crossover logbook
        for cross_hash in self.crossover_logbook.chapters.keys():

            # Unpack the crossover log into a dictionary
            cross_log = self.crossover_logbook.chapters[cross_hash]
            cross_results = {
                "gen": self.crossover_logbook.select("gen"),
                "called": cross_log.select("called"),
                "failed": cross_log.select("failed"),
                "survivor": cross_log.select("survivor"),
            }

            # Append the results to the crossover dictionary
            crossover_dict["results"].append(cross_results)

        # Save the crossover dictionary to a json file
        file_path = os.path.join(save_dir, "crossovers.json")
        with open(file_path, "w") as f:
            json.dump(crossover_dict, f, indent=4)

    def _save_mutations(self, save_dir: str):
        """Saves the mutation logbook and information to a json file.

        The json file will contain the following information:

        - 'names': The names of the mutations
        - 'weights': The weights of the mutations
        - 'reprs': The representations of the mutations
        - 'hashes': The hashes of the mutations
        - 'results': A list of dictionaries containing the following information:

            - 'gen': The generation number
            - 'called': The number of times the mutation was called
            - 'failed': The number of times the mutation failed
            - 'survivor': The number of times the mutant was selected as a survivor

        :param save_dir: The directory to save the file to.
        """
        # Set up a dictionary to store mutation information and results
        mutation_dict = {
            "names": [mut.__class__.__name__ for mut in self.ga_runner.mutation_list],
            "weights": list(self.ga_runner.norm_mutation_weights),
            "reprs": [mut.__repr__() for mut in self.ga_runner.mutation_list],
            "hashes": [mut.__hash__() for mut in self.ga_runner.mutation_list],
            "results": [],
        }
        # Loop over all chapters in the mutation logbook
        for mut_hash in self.mutation_logbook.chapters.keys():

            # Unpack the mutation log into a dictionary
            mut_log = self.mutation_logbook.chapters[mut_hash]
            mut_results = {
                "gen": self.mutation_logbook.select("gen"),
                "called": mut_log.select("called"),
                "failed": mut_log.select("failed"),
                "survivor": mut_log.select("survivor"),
            }

            # Append the results to the mutation dictionary
            mutation_dict["results"].append(mut_results)

        # Save the mutation dictionary to a json file
        file_path = os.path.join(save_dir, "mutations.json")
        with open(file_path, "w") as f:
            json.dump(mutation_dict, f, indent=4)

    def _save_fitnesses(self, save_dir: str):
        """Saves the fitness logbook and information to a json file.

        The json file will contain the following information:

        - 'names': The names of the fitness functions
        - 'weights': The weights of the fitness functions
        - 'reprs': The representations of the fitness functions
        - 'titles': The `db_title` of the fitness functions
        - 'hashes': The hashes of the fitness functions
        - 'results': A list of dictionaries containing the following information:

            - 'gen': The generation number
            - 'min': The minimum fitness value
            - 'max': The maximum fitness value
            - 'avg': The average fitness value
            - 'std': The standard deviation of the fitness values

        :param save_dir: The directory to save the file to.
        """
        # Set up a dictionary to store fitness information and results
        fitnesses_dict = {
            "names": [
                func.__class__.__name__ for func in self.ga_runner.fitness_functions
            ],
            "weights": list(self.ga_runner.fitness_weights),
            "reprs": [func.__repr__() for func in self.ga_runner.fitness_functions],
            "titles": [func.db_title for func in self.ga_runner.fitness_functions],
            "hashes": [func.__hash__() for func in self.ga_runner.fitness_functions],
            "results": [],
        }
        # Loop over all chapters in the fitness logbook
        for func_hash in self.fitness_logbook.chapters.keys():

            # Unpack the fitness log into a dictionary
            func_log = self.fitness_logbook.chapters[func_hash]
            func_results = {
                "gen": self.fitness_logbook.select("gen"),
                "min": func_log.select("min"),
                "max": func_log.select("max"),
                "avg": func_log.select("avg"),
                "std": func_log.select("std"),
            }

            # Append the results to the fitness dictionary
            fitnesses_dict["results"].append(func_results)

        # Save the fitness dictionary to a json file
        file_path = os.path.join(save_dir, "fitnesses.json")
        with open(file_path, "w") as f:
            json.dump(fitnesses_dict, f, indent=4)

    @property
    def info_dict(self) -> dict:
        info_dict = {}
        info_dict["break_condition"] = self.ga_runner.break_condition.__repr__()
        # Get the current number of generations. Needs to be called after the
        # run method.
        info_dict["n_generations"] = self.ga_runner.generation
        info_dict["parent_selection"] = self.ga_runner.parent_selection.__repr__()
        info_dict["parent_ratio"] = self.ga_runner.parent_ratio
        info_dict["survivor_selection"] = self.ga_runner.survivor_selection.__repr__()

        return info_dict

    def run(
        self,
        population: Population,
        global_log: tools.Logbook,
        global_stats: tools.MultiStatistics | None,
    ) -> Population:
        assert hasattr(self, "id"), "Stage ID not set."

        # Attach the logger to the ga_runner
        self.ga_runner.logger = self.logger

        population = self.ga_runner.run(
            population=population,
            global_log=global_log,
            global_stats=global_stats,
            stage_id=self.id,
            stop_event=self.stop_event,
        )

        self.crossover_logbook = self.ga_runner.crossover_logbook
        self.mutation_logbook = self.ga_runner.mutation_logbook
        self.fitness_logbook = self.ga_runner.fitness_logbook
        self.hall_of_fame = self.ga_runner.hall_of_fame

        return population

    def save_results(
        self,
        save_dir: str,
        structures_db: Database,
        global_statistics_dict: dict[str, Callable[[Individual], float]] | None = None,
    ):
        self._save_mutations(save_dir)
        self._save_crossovers(save_dir)
        self._save_fitnesses(save_dir)
        self._save_hall_of_fame(
            structures_db,
            self.hall_of_fame,
            self.ga_runner.fitness_functions,
            global_statistics_dict,
        )

    def with_same_config(self, rng: None | np.random.Generator = None) -> Stage:
        """New GAStage with identical configuration and no accumulated run state.

        Note: References to initial objects stay the same and no deep copy is performed.
        If any of the objects cannot handle this, please manually create copy.
        """
        new = GAStage(**self._cfg, rng=rng)  # type: ignore
        return new
