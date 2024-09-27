from enum import global_str
from tqdm.gui import tqdm
from fucrimodo.customs.crossovers import Crossover

from .modules import Mutation, FitnessFunction, BreakCondition
from .utils import data_handeling
from .ga import myEaSimple
from .utils.custom_soap import CustomSOAP
import numpy as np
import random
from deap import base, creator, tools, algorithms
import ase
from typing import Callable
import functools
from icecream import ic
from copy import deepcopy


# ╔══════════════════════════════════════════════════════════╗
# ║                    Utility functions                     ║
# ╚══════════════════════════════════════════════════════════╝

def get_unique_keys(
    keys: list[str],
) -> list[str]:
    """
    Checks if provided keys are unique. If not will append a letter to the key.

    :param keys: List of keys that should be unique

    :return: List of unique keys

    :raises ValueError: If too many keys with the same name exit (more than 10)
    """
    key_append_list = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J"
    ]
    unique_keys = []
    for key in keys:

        new_key = key 
        rerun_step = 0
        while new_key in unique_keys:
            new_key = key + "_" + key_append_list[rerun_step]
            rerun_step += 1
            if rerun_step == len(key_append_list):
                raise ValueError(
                    "Too many fitness functions with the same name."
                )

        unique_keys.append(new_key)

    return unique_keys


#  ╒══════════════════════════════════════════════════════════╕
#                     Grid Search Functions
#  ╘══════════════════════════════════════════════════════════╛

class Stage:
    """
    Class for creating and running a GeneticAlgorithm.
    The parameters for the GeneticAlgorithm are set in the params dict.
    The init function checks if all parameters are set.
    """

    def __init__(self, stage_data: data_handeling.StageData) -> None:
        self.stage_data = stage_data

    def __create_global_statistics(
        self, global_stats_dict: dict[str, Callable[[ase.Atoms], float]]
    ) -> tuple[tools.MultiStatistics, list[str]]:
        capter_keys = []
        stats_dict = {}

        for key, func in global_stats_dict.items():
            stats_dict[key] = tools.Statistics(
                key=func
            )
            capter_keys.append(key)

        mstats = tools.MultiStatistics(**stats_dict)
        mstats.register("avg", np.mean)
        mstats.register("max", np.max)
        mstats.register("min", np.min)
        mstats.register("std", np.std)

        return mstats, capter_keys

    def __create_fitness_statistics(
        self
    ) -> tuple[tools.MultiStatistics, list[str]]:
        """
        Creates the :class:`deap.tools.MultiStatistics` objects to track statistics.

        Uses the :attr:`FitnessFunction.db_titles` of each fitness of the 
        stage to set the chapter of the :class:`deap.tools.MultiStatistics` 
        object.
        If a name is set multiple times, a letter is appended to the name.
        For the fitness functions and global statistics the mean, max, min
        and std values are tracked for each generation.

        :param fitness_functions: List of the fitness functions that are used
            during the stage

        :return: A tuple that contains the MultiStatistics object and a list of 
            the chapter keys that are set for the fitness functions.
        """
        capter_keys = []
        fitness_stats_dict = {}

        # Uses the index of the fitness to get the specific value that is stored
        # in the individual.
        def get_specific_fit_val(ind, index):
            return ind.fitness.values[index]

        unique_fitness_names = get_unique_keys([
            fitness_function.get_db_title() 
            for fitness_function in self.stage_data.fitness_functions
        ])

        for i, name in enumerate(unique_fitness_names):
            # use partial to prevent lambda usage in loop,
            # because lamda in loop is not good
            key_func = functools.partial(get_specific_fit_val, index=i)
            fitness_stats_dict[name] = tools.Statistics(
                key=key_func
            )
            capter_keys.append(name)

        mstats = tools.MultiStatistics(
            **fitness_stats_dict
        )
        mstats.register("avg", np.mean)
        mstats.register("max", np.max)
        mstats.register("min", np.min)
        mstats.register("std", np.std)

        return mstats, unique_fitness_names

    def create_gen_alg_toolbox(
        self,
        fitness_functions: list[FitnessFunction],
        fitness_weights: tuple[float, ...],
        crossover_list: list[Crossover],
        crossover_weights: list[float] | list[int],
        mutation_list: list[Mutation],
        mutation_weights: list[float] | list[int],
        break_condition: BreakCondition
    ) -> base.Toolbox:

        creator.create(
            "FitnessMulti", base.Fitness, weights=fitness_weights
        )
        creator.create(
            "Individual",
            ase.Atoms,
            fitness=creator.FitnessMulti  # type:ignore
        )

        def evaluate(individual: ase.Atoms) -> tuple[float, ...]:
            fitness_tuple = ()
            for fitness_function in fitness_functions:
                fitness_tuple += (
                    fitness_function.evaluate_individual(individual),
                )
            return fitness_tuple

        def crossover(
            parent1: ase.Atoms,
            parent2: ase.Atoms
        ) -> tuple[ase.Atoms, ase.Atoms, bool, str]:
            selected_crossover = random.choices(
                crossover_list,
                weights=crossover_weights
            )[0]

            if hasattr(selected_crossover, "__repr__"):
                crossover_name = selected_crossover.__repr__()
            else:
                crossover_name = selected_crossover.__class__.__name__

            return selected_crossover.crossover(parent1, parent2) + \
                (crossover_name,)

        def mutate(individual: ase.Atoms) -> tuple[ase.Atoms, bool, str]:
            selected_mutation = random.choices(
                mutation_list,
                weights=mutation_weights
            )[0]

            if hasattr(selected_mutation, "__repr__"):
                mutation_name = selected_mutation.__repr__()
            else:
                mutation_name = selected_mutation.__class__.__name__

            return selected_mutation.mutate(individual) + \
                (mutation_name,)

        toolbox = base.Toolbox()

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate)
        # Changed from tournsize=3 to tournsize=5
        toolbox.register("select_parents", tools.selTournament, tournsize=4)
        # toolbox.register("select_parents", tools.selNSGA2)
        toolbox.register("select_survivors", tools.selNSGA2)
        toolbox.register("break_condition", break_condition.check)

        return toolbox

    def get_hall_of_fame_data(
        self,
        hall_of_fame: tools.HallOfFame,
        fitness_functions: list[FitnessFunction],
        fitness_keys: list[str],
        global_statistics_dict: dict[str, Callable[[ase.Atoms], float]] | None = None,
        global_stats_keys: list[str] | None = None
    ) -> tuple[list[ase.Atoms], list[dict]]:
        """
        Returns the best crystal and the key value pairs for the best crystal.
        """
        hof_crystals = [ind for ind in hall_of_fame]
        hof_key_value_pairs_list = []

        for crystal in hof_crystals:
            key_value_pairs = {}
            key_value_pairs["type"] = "hof"

            for i, fitness_function in enumerate(fitness_functions):
                key_value_pairs[fitness_keys[i]] = fitness_function.evaluate_individual(crystal)

            if global_statistics_dict is not None and global_stats_keys is not None:
                for i, func in enumerate(global_statistics_dict.values()):
                    key_value_pairs[global_stats_keys[i]] = func(crystal)

            hof_key_value_pairs_list.append(key_value_pairs)

        return hof_crystals, hof_key_value_pairs_list

    def get_best_crystals_of_last_generation(
        self,
        population: list[ase.Atoms],
        fitness_functions: list[FitnessFunction],
        add_stats_func: Callable[[ase.Atoms], float] | None = None,
        add_stats_func_name: str | None = None
    ) -> tuple[list[ase.Atoms], list[dict]]:
        """
        Returns the best crystal and the key value pairs for the best crystal.
        """
        best_crystals = tools.selBest(
            individuals=population,
            k=self.stage_data.save_n_best_crystals
        )
        best_key_value_pairs_list = []

        for crystal in best_crystals:
            key_value_pairs = {}
            key_value_pairs["type"] = "best_of_last_gen"

            for fitness_function in fitness_functions:
                key_value_pairs[fitness_function.get_db_title()] = (
                    fitness_function.evaluate_individual(crystal)
                )

            if add_stats_func is not None:
                if add_stats_func_name is None:
                    add_stats_func_name = add_stats_func.__name__
                value = add_stats_func(crystal)
                key_value_pairs[add_stats_func_name] = value

            best_key_value_pairs_list.append(key_value_pairs)

        return best_crystals, best_key_value_pairs_list

    def run_with_set_params(
        self,
        n_generations: int,
        start_pop: list[ase.Atoms],
        adjust_fitness_functions: bool = True,
        title: str = "Gen Alg",
        soap_obj: CustomSOAP | None = None,
        global_statistics_dict: dict[str, Callable[[ase.Atoms], float]] | None = None,
    ) -> list[ase.Atoms]:
        """This is the main function of this class.
        It runs the GeneticAlgorithm with the set parameters.
        It then analyses the gen alg run and adds the data to the
        saves the data with the stage_data.
        """
        print("Running Genetic Algorithm...")
        print("Number of Generations:{}".format(n_generations), end="\n")
        print("Size of Start Population:{}".format(len(start_pop)), end="\n\n")

        if adjust_fitness_functions:
            # Create a deep copy of the fitness functions to prevent
            # changes to the original fitness functions for save in file
            self.stage_data.fitness_functions = [
                deepcopy(fitness_function)
                for fitness_function in self.stage_data.fitness_functions
            ]

            for fitness_function in self.stage_data.fitness_functions:
                fitness_function.adjust_to_population(start_pop)

        toolbox = self.create_gen_alg_toolbox(
            fitness_functions=self.stage_data.fitness_functions,
            fitness_weights=self.stage_data.fitness_weights,
            crossover_list=self.stage_data.crossover_list,
            crossover_weights=self.stage_data.crossover_weights,
            mutation_list=self.stage_data.mutation_list,
            mutation_weights=self.stage_data.mutation_weights,
            break_condition=self.stage_data.break_condition
        )

        fitness_stats, fitness_keys = self.__create_fitness_statistics()
        if global_statistics_dict is not None:
            global_stats, global_stats_keys = self.__create_global_statistics(
                global_stats_dict=global_statistics_dict
            )
        else:
            global_stats = None
            global_stats_keys = None

        hall_of_fame = tools.HallOfFame(self.stage_data.save_n_best_crystals)

        population = [
            creator.Individual(ind) for ind in start_pop  # type: ignore
        ]

        mut_log = {}
        for mutation in self.stage_data.mutation_list:
            mut_log[mutation.__repr__()] = {"called": [], "failed": []}
        cross_log = {}
        for cross in self.stage_data.crossover_list:
            cross_log[cross.__repr__()] = {"called": [], "failed": []}

        pop, fitness_logbook, global_logbook, cross_log, mut_log = myEaSimple(
            population=population,
            toolbox=toolbox,
            cxpb=self.stage_data.crossover_probability,
            mutpb=self.stage_data.mutation_probability,
            ngen=self.stage_data.n_generations,
            mutation_log=mut_log,
            crossover_log=cross_log,
            fitness_stats=fitness_stats,
            global_stats=global_stats,
            halloffame=hall_of_fame,
            verbose=True,
            progress_bar_title=title,
            soap_obj=soap_obj,
        )

        self.stage_data.save_log(
            fitness_logbook=fitness_logbook,
            global_logbook=global_logbook,
            crossover_log=cross_log,
            mutation_log=mut_log
        )

        hof_crystals, hof_key_value_pairs_list = self.get_hall_of_fame_data(
            hall_of_fame=hall_of_fame,
            fitness_functions=self.stage_data.fitness_functions,
            fitness_keys=fitness_keys,
            global_statistics_dict=global_statistics_dict,
            global_stats_keys=global_stats_keys
        )

        # best_crystals, best_key_value_pairs_list = (
        #     self.get_best_crystals_of_last_generation(
        #         population=pop,
        #         fitness_functions=self.stage_data.fitness_functions,
        #         global_statistics_dict=global_statistics_dict,
        #     )
        # )

        self.stage_data.save_crystals_in_db(
            hof_crystals, hof_key_value_pairs_list
        )

        return pop


class MultiGenAlgSearch:
    def __init__(
        self,
        run_data: data_handeling.RunData,
    ) -> None:
        self.run_data = run_data

    def run(
        self, start_pop_candidates: list[ase.Atoms]
    ) -> None:
        print("Starting Multi Gen Alg Search...")
        print(f"Total number of searches: {self.run_data.n_stages}\n\n")

        individuals = start_pop_candidates

        self.run_data.add_start_time()
        for i in range(self.run_data.n_stages):
            id = i + 1

            # print("Running Stage: {}/{}".format(
            #     id, self.run_data.n_stages
            # ), end="\n\n")

            config_data = self.run_data.get_stage_data(id)
            start_pop = config_data.start_population_selection.select_start_pop(
                individuals=individuals
            )

            config = Stage(config_data)
            final_pop = config.run_with_set_params(
                n_generations=config_data.n_generations,
                start_pop=start_pop,
                title=f"Stage {id}/{self.run_data.n_stages}. Gen Alg",
                soap_obj=self.run_data.soap_object,
                global_statistics_dict = self.run_data.global_statistics_dict
            )

            individuals = final_pop

        # NOTE: The following should not be here but be in the main file!
        print()
        print("Multi Gen Alg Search finished.")
        print(f"Saving run data at {self.run_data.run_dir}", end="\n\n")
        self.run_data.add_end_time()
        self.run_data.save_run_info_json()
