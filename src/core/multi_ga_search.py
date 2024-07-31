from tqdm.gui import tqdm
from src.fucrimodo.crystal_creation.crossovers import Crossover
from src.fucrimodo.crystal_creation.mutations import Mutation
from ..utils import data_handeling
from src.fucrimodo.crystal_creation.fitness_functions import FitnessFunction
from src.fucrimodo.crystal_creation.ga import myEaSimple
from src.fucrimodo.utils.custom_soap import CustomSOAP
import numpy as np
import random
from deap import base, creator, tools, algorithms
import ase
from typing import Callable
import functools
from icecream import ic
from copy import deepcopy
from src.fucrimodo.crystal_creation import break_conditions as break_cond


# ╔══════════════════════════════════════════════════════════╗
# ║                    Utility functions                     ║
# ╚══════════════════════════════════════════════════════════╝

def make_fitness_function_titles_unique(
    fitness_functions: list[FitnessFunction],
) -> None:
    """
    Checks if names of fitness functions are unique.
    If not, appends a letter to the name to make it unique.
    New name is then saved as db_title in the fitness function.
    This is done to prevent overwriting of data in the database.
    """
    title_append_list = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J"
    ]
    unique_titles = []
    for i, fitness_function in enumerate(fitness_functions):
        title = fitness_function.get_db_title()
        step = 0

        new_title = title
        while new_title in unique_titles:
            new_title = title + "_" + title_append_list[i]
            step += 1
            if step == len(title_append_list):
                raise ValueError(
                    "Too many fitness functions with the same name."
                )

        if new_title != title:
            fitness_function.set_db_title(new_title)
            unique_titles.append(new_title)
        else:
            unique_titles.append(title)


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

    def create_stats(
        self,
        fitness_functions: list[FitnessFunction],
        add_stats_func: Callable[[ase.Atoms], float] | None = None,
        add_stats_func_name: str | None = None
    ) -> tuple[tools.MultiStatistics, list[str]]:
        """
        Creates a statistics object for the Genetic Algorithm.
        Automatically registers the mean, max and min fitness.
        For an additional evaluation function the mean, max and min value is
        also registered.
        Returns the statistics object and a list of the header names.
        """
        capter_keys = []
        fitness_stats_dict = {}

        def get_specific_fit_val(ind, index):
            return ind.fitness.values[index]

        fitness_stats_dict["TotalFitness"] = tools.Statistics(
            key=lambda ind: ind.fitness.values
        )
        capter_keys.append("TotalFitness")

        make_fitness_function_titles_unique(fitness_functions)

        for i, fitness_function in enumerate(fitness_functions):
            name = fitness_function.get_db_title()

            # use partial to prevent lambda usage in loop,
            # because lamda in loop is not good
            key_func = functools.partial(get_specific_fit_val, index=i)
            fitness_stats_dict[name] = tools.Statistics(
                key=key_func
            )
            capter_keys.append(name)

        if add_stats_func is not None:
            stats = tools.Statistics(key=lambda ind: add_stats_func(ind))
            func_name = add_stats_func_name
            if func_name is None:
                func_name = add_stats_func.__name__

            ic("Adding fitness_stats_dict to MultiStatistics")
            ic(f"fitness_stats_dict: {fitness_stats_dict}")
            ic(f"chapter_keys: {capter_keys}")
            mstats = tools.MultiStatistics(
                **fitness_stats_dict,
                **{func_name: stats}
            )
            capter_keys.append(func_name)

        else:
            ic("Adding fitness_stats_dict to MultiStatistics")
            ic(f"fitness_stats_dict: {fitness_stats_dict}")
            ic(f"chapter_keys: {capter_keys}")
            mstats = tools.MultiStatistics(
                **fitness_stats_dict
            )

        mstats.register("avg", np.mean)
        mstats.register("max", np.max)
        mstats.register("min", np.min)
        mstats.register("std", np.std)

        return mstats, capter_keys

    def create_gen_alg_toolbox(
        self,
        fitness_functions: list[FitnessFunction],
        fitness_weights: tuple[float, ...],
        crossover_list: list[Crossover],
        crossover_weights: list[float] | list[int],
        mutation_list: list[Mutation],
        mutation_weights: list[float] | list[int],
        break_condition: break_cond.BreakCondition
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
        ) -> tuple[ase.Atoms, ase.Atoms, bool]:
            selected_crossover = random.choices(
                crossover_list,
                weights=crossover_weights
            )[0]
            return selected_crossover.crossover(parent1, parent2)

        def mutate(individual: ase.Atoms) -> tuple[ase.Atoms, bool]:
            selected_mutation = random.choices(
                mutation_list,
                weights=mutation_weights
            )[0]
            return selected_mutation.mutate(individual)

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
        add_stats_func: Callable[[ase.Atoms], float] | None = None,
        add_stats_func_name: str | None = None
    ) -> tuple[list[ase.Atoms], list[dict]]:
        """
        Returns the best crystal and the key value pairs for the best crystal.
        """
        hof_crystals = [ind for ind in hall_of_fame]
        hof_key_value_pairs_list = []

        for crystal in hof_crystals:
            key_value_pairs = {}
            key_value_pairs["type"] = "hof"

            for fitness_function in fitness_functions:
                key_value_pairs[fitness_function.get_db_title()] = (
                    fitness_function.evaluate_individual(crystal)
                )

            if add_stats_func is not None:
                if add_stats_func_name is None:
                    add_stats_func_name = add_stats_func.__name__
                value = add_stats_func(crystal)
                key_value_pairs[add_stats_func_name] = value

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
    ) -> list[ase.Atoms]:
        """
        This is the main function of this class.
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

        mstats, log_chapter_keys = self.create_stats(
            fitness_functions=self.stage_data.fitness_functions,
            add_stats_func=self.stage_data.additional_statistics_func,
            add_stats_func_name=self.stage_data.add_stats_func_name
        )

        hall_of_fame = tools.HallOfFame(self.stage_data.save_n_best_crystals)

        population = [
            creator.Individual(ind) for ind in start_pop  # type: ignore
        ]

        pop, log = myEaSimple(
            population=population,
            toolbox=toolbox,
            cxpb=self.stage_data.crossover_probability,
            mutpb=self.stage_data.mutation_probability,
            ngen=self.stage_data.n_generations,
            stats=mstats,
            halloffame=hall_of_fame,
            verbose=True,
            progress_bar_title=title,
            soap_obj=soap_obj,
        )

        self.stage_data.save_log(log, log_chapter_keys)

        hof_crystals, hof_key_value_pairs_list = self.get_hall_of_fame_data(
            hall_of_fame=hall_of_fame,
            fitness_functions=self.stage_data.fitness_functions,
            add_stats_func=self.stage_data.additional_statistics_func,
            add_stats_func_name=self.stage_data.add_stats_func_name
        )

        best_crystals, best_key_value_pairs_list = (
            self.get_best_crystals_of_last_generation(
                population=pop,
                fitness_functions=self.stage_data.fitness_functions,
                add_stats_func=self.stage_data.additional_statistics_func,
                add_stats_func_name=self.stage_data.add_stats_func_name
            )
        )

        self.stage_data.save_crystals_in_db(
            hof_crystals + best_crystals,
            hof_key_value_pairs_list + best_key_value_pairs_list
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
            start_pop = config_data.start_pop_generation.select_start_pop(
                individuals=individuals
            )

            config = Stage(config_data)
            final_pop = config.run_with_set_params(
                n_generations=config_data.n_generations,
                start_pop=start_pop,
                title=f"Stage {id}/{self.run_data.n_stages}. Gen Alg",
                soap_obj=self.run_data.soap_object
            )

            individuals = []
            for ind in final_pop:
                # # Remove fitness values from individuals for next run
                # if hasattr(ind, "fitness"):
                #     del ind.fitness  # type: ignore

                individuals.append(ind)

        # NOTE: The following should not be here but be in the main file!
        print()
        print("Multi Gen Alg Search finished.")
        print(f"Saving run data at {self.run_data.run_dir}", end="\n\n")
        self.run_data.add_end_time()
        self.run_data.save_run_info_json()
