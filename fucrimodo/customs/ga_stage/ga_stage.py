from collections.abc import Callable
import functools
from fucrimodo.core.modules import Stage, Population, FitnessFunction, PopulationSelection, Individual
from .mutations import Mutation
from .crossovers import Crossover
from .break_conditions import BreakCondition
from ase.db.core import Database
from typing import Any, Sequence
from deap import tools
import numpy as np
import os
from .genetic_algorithm import GeneticAlgorithm

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
        description: str = "",
        n_crystals_to_save: int = 10,
        parent_selection: Callable = functools.partial(
            tools.selTournament, tournsize=5
        ),
        survivor_selection: Callable = tools.selNSGA2,
    ):
        super().__init__(name, description)

        fitness_funcs, fitness_weights = self.__seperate_object_weight_tuples(
            fitness_functions
        )
        cross_list, crossover_weights = self.__seperate_object_weight_tuples(
            crossover_list
        )
        mut_list, mutation_weights = self.__seperate_object_weight_tuples(
            mutation_list
        )

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
            parent_selection=parent_selection,
            survivor_selection=survivor_selection,
        )

    def __seperate_object_weight_tuples(
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
                weights += (1.,)

        return objects, weights

    def run(
        self, 
        population: Population,
        global_log: tools.Logbook,
        global_stats: tools.MultiStatistics | None,
    ) -> Population:
        assert hasattr(self, "id"), "Stage ID not set."

        population = self.ga_runner.run(
            population=population, 
            global_log=global_log,
            global_stats=global_stats,
            stage_id=self.id
        )

        self.crossover_logbook = self.ga_runner.crossover_logbook
        self.mutation_logbook = self.ga_runner.mutation_logbook
        self.fitness_logbook = self.ga_runner.fitness_logbook

        return population

    def save_results(self, save_path: str, crystals_db: Database):
        import pickle

        file_path = os.path.join(save_path, f"stage_{self.id}_crossover.pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.crossover_logbook, f)

        file_path = os.path.join(save_path, f"stage_{self.id}_mutation.pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.mutation_logbook, f)

        file_path = os.path.join(save_path, f"stage_{self.id}_fitness.pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.fitness_logbook, f)
