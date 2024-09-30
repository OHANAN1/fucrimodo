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

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


    # def save_crystals_in_db(
    #     self,
    #     crystals: list[ase.Atoms],
    #     key_value_pairs_list: list[dict],
    # ) -> None:
    #     """
    #     Saves the most similar crystals of the stage in the crystal
    #     database of the run.
    #     The tuple contains the crystal and the key value pairs of the crystal.
    #     Also adds the stage id to the key value pairs.
    #     """
    #     i = 0
    #     for crystal, key_value_pairs_dict in zip(
    #         crystals, key_value_pairs_list
    #     ):
    #         key_value_pairs_dict["stage_id"] = self.stage_id
    #         self.crystal_database.write(
    #             crystal,
    #             key_value_pairs_dict
    #         )
    #         i += 1
    #
    # def __unpack_logbook(
    #     self, 
    #     logbook: tools.Logbook, 
    #     value_types: list[str] = ["min", "max", "avg", "std"],
    # ) -> dict:
    #     log_dict = {}
    #     for key in logbook.chapters.keys():
    #         log_dict[key] = {}
    #         for value_type in value_types:
    #             log_dict[key][value_type] = logbook.chapters[key].select(
    #                 value_type
    #             )
    #
    #     return log_dict
    #
    # def save_log(
    #     self,
    #     mutation_log: dict[str, dict[str, list[int]]],
    #     crossover_log: dict[str, dict[str, list[int]]],
    #     fitness_logbook: tools.Logbook,
    #     global_logbook: tools.Logbook | None = None,
    # ) -> None:
    #     fitness_log_dict = self.__unpack_logbook(
    #         logbook = fitness_logbook
    #     )
    #
    #     global_log_dict = {}
    #     if global_logbook is not None:
    #         global_log_dict = self.__unpack_logbook(
    #             logbook = global_logbook
    #         )
    #
    #     self.save_file_path = f"{self.run_dir}/stage_{self.stage_id}.json"
    #     with open(self.save_file_path, "w") as f:
    #         json.dump(
    #             {
    #                 "fitness_log": fitness_log_dict,
    #                 "global_statistics_log": global_log_dict,
    #                 "mutation_data": mutation_log,
    #                 "crossover_data": crossover_log,
    #             },
    #             f, indent=4, default=convert_to_serializable
    #         )

class GAStage(Stage):
    def __init__(
        self, 
        id: int,
        fitness_functions: Sequence[FitnessFunction | tuple[FitnessFunction, float]],
        crossover_list: Sequence[Crossover | tuple[Crossover, float]],
        mutation_list: Sequence[Mutation | tuple[Mutation, float]],
        mutation_probability: float,
        crossover_probability: float,
        break_condition: BreakCondition,
        n_crystals_to_save: int = 10,
        parent_selection: Callable = functools.partial(
            tools.selTournament, tournsize=5
        ),
        survivor_selection: Callable = tools.selNSGA2,
    ):
        super().__init__(id)

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
            stage_id=self.id,
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

        population = self.ga_runner.run(
            population, global_stats, global_log
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
