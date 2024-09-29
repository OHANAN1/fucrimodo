from collections.abc import Callable
import functools
from fucrimodo.core.modules import Stage, Population, FitnessFunction, PopulationSelection, Individual
from .mutations import Mutation
from .crossovers import Crossover
from .break_conditions import BreakCondition
from ase.db.core import Database
import ase
from typing import Any, Sequence
from deap import tools, creator, base
import json
import numpy as np
import random
from icecream import ic
import os

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class StageData(dict):
    """
    Class for storing and saving all the data about a stage.
    Gets initialized and handled by the RunData class.
    Contains paremeters specific to the stage
    and the hyperparameters for th gridsearch needed to run the stage.
    Stores the results of the structure search.
    """

    def __init__(
        self,
        number_of_generations: int,
        fitness_tuples: Sequence[FitnessFunction | tuple[FitnessFunction, float]],
        crossover_tuples: Sequence[Crossover | tuple[Crossover, float]], 
        crossover_prob: float,
        mutation_tuples: Sequence[Mutation | tuple[Mutation, float]],
        mutation_prob: float,
        break_condition: BreakCondition,
        parent_selection: Callable,
        survivor_selection: Callable,
        n_crystals_to_save: int,
    ) -> None:
        self.fitness_functions = fitness_tuples
        self.fitness_weights = fitness_tuples

        self.mutation_list = mutation_tuples
        self.mutation_weights = mutation_tuples

        self.crossover_list = crossover_tuples
        self.crossover_weights = crossover_tuples

        self.mutation_probability = mutation_prob
        self.crossover_probability = crossover_prob
        self.break_condition = break_condition
        self.n_generations = number_of_generations
        self._hall_of_fame = tools.HallOfFame(n_crystals_to_save)

        self.survivor_selection = survivor_selection
        self.parent_selection = parent_selection

    @property
    def fitness_functions(self) -> list[FitnessFunction]:
        return self._fitness_functions

    @fitness_functions.setter
    def fitness_functions(
        self, 
        fitness_tuples: Sequence[FitnessFunction | tuple[FitnessFunction, float]]
    ):
        fit_funcs, _ = self.__seperate_object_weight_tuples(fitness_tuples)
        self._fitness_functions = fit_funcs

    @property
    def fitness_weights(self) -> tuple:
        return self._fitness_weights

    @fitness_weights.setter
    def fitness_weights(
        self, 
        fitness_tuples: Sequence[FitnessFunction | tuple[FitnessFunction, float]]
    ):
        _, weights = self.__seperate_object_weight_tuples(fitness_tuples)
        self._fitness_weights = weights

    @property
    def mutation_list(self) -> list[Mutation]:
        return self._mutation_list

    @mutation_list.setter
    def mutation_list(
        self, 
        mutation_tuples: Sequence[Mutation | tuple[Mutation, float]]
    ):
        mut_list, _ = self.__seperate_object_weight_tuples(mutation_tuples)
        self._mutation_list = mut_list

    @property
    def mutation_weights(self) -> tuple:
        return self._mutation_weights

    @mutation_weights.setter
    def mutation_weights(
        self, 
        mutation_tuples: Sequence[Mutation | tuple[Mutation, float]]
    ):
        _, weights = self.__seperate_object_weight_tuples(mutation_tuples)
        self._mutation_weights = weights

    @property
    def crossover_list(self) -> list[Crossover]:
        return self._crossover_list

    @crossover_list.setter
    def crossover_list(
        self, 
        crossover_tuples: Sequence[Crossover | tuple[Crossover, float]]
    ):
        cross_list, _ = self.__seperate_object_weight_tuples(crossover_tuples)
        self._crossover_list = cross_list

    @property
    def crossover_weights(self) -> tuple:
        return self._crossover_weights

    @crossover_weights.setter
    def crossover_weights(
        self, 
        crossover_tuples: Sequence[Crossover | tuple[Crossover, float]]
    ):
        _, weights = self.__seperate_object_weight_tuples(crossover_tuples)
        self._crossover_weights = weights

    @property
    def hall_of_fame(self) -> tools.HallOfFame:
        return self._hall_of_fame

    @hall_of_fame.setter
    def hall_of_fame(self, hall_of_fame: tools.HallOfFame):
        self._hall_of_fame = hall_of_fame

    def add_run_settings(
        self,
        crystal_database: Database,
        run_dir: str,
        stage_id: int,
        save_n_best_crystals: int,
    ) -> None:
        """
        Adds the run settings to the stage data.
        Gets called by the RunData class automatically.
        """
        self.run_dir = run_dir
        self.stage_id = stage_id
        self.crystal_database = crystal_database
        self.save_n_best_crystals = save_n_best_crystals

    def get_params_dict(self) -> dict:
        """
        Returns the parameters of the stage as a dictionary.
        """
        params_dict = {
            "number of generations": self.n_generations,
            "fitness functions": self.fitness_functions,
            "fitness weights": self.fitness_weights,
            "crossover list": self.crossover_list,
            "crossover weights": self.crossover_weights,
            "crossover probability": self.crossover_probability,
            "mutation list": self.mutation_list,
            "mutation weights": self.mutation_weights,
            "mutation probability": self.mutation_probability,
            "break condition": self.break_condition,
        }
        return params_dict

    def save_crystals_in_db(
        self,
        crystals: list[ase.Atoms],
        key_value_pairs_list: list[dict],
    ) -> None:
        """
        Saves the most similar crystals of the stage in the crystal
        database of the run.
        The tuple contains the crystal and the key value pairs of the crystal.
        Also adds the stage id to the key value pairs.
        """
        i = 0
        for crystal, key_value_pairs_dict in zip(
            crystals, key_value_pairs_list
        ):
            key_value_pairs_dict["stage_id"] = self.stage_id
            self.crystal_database.write(
                crystal,
                key_value_pairs_dict
            )
            i += 1

    def __unpack_logbook(
        self, 
        logbook: tools.Logbook, 
        value_types: list[str] = ["min", "max", "avg", "std"],
    ) -> dict:
        log_dict = {}
        for key in logbook.chapters.keys():
            log_dict[key] = {}
            for value_type in value_types:
                log_dict[key][value_type] = logbook.chapters[key].select(
                    value_type
                )

        return log_dict

    def save_log(
        self,
        mutation_log: dict[str, dict[str, list[int]]],
        crossover_log: dict[str, dict[str, list[int]]],
        fitness_logbook: tools.Logbook,
        global_logbook: tools.Logbook | None = None,
    ) -> None:
        fitness_log_dict = self.__unpack_logbook(
            logbook = fitness_logbook
        )

        global_log_dict = {}
        if global_logbook is not None:
            global_log_dict = self.__unpack_logbook(
                logbook = global_logbook
            )

        self.save_file_path = f"{self.run_dir}/stage_{self.stage_id}.json"
        with open(self.save_file_path, "w") as f:
            json.dump(
                {
                    "fitness_log": fitness_log_dict,
                    "global_statistics_log": global_log_dict,
                    "mutation_data": mutation_log,
                    "crossover_data": crossover_log,
                },
                f, indent=4, default=convert_to_serializable
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


class GAStage(Stage):
    def __init__(
        self, 
        id: int,
        number_of_generations: int,
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
        self.stage_data = StageData(
            number_of_generations=number_of_generations,
            fitness_tuples=fitness_functions,
            crossover_tuples=crossover_list,
            crossover_prob=crossover_probability,
            mutation_tuples=mutation_list,
            mutation_prob=mutation_probability,
            break_condition=break_condition,
            n_crystals_to_save=n_crystals_to_save,
            parent_selection=parent_selection,
            survivor_selection=survivor_selection,
        )

        self._fitness_stats = None

    @property
    def fitness_stats(self) -> tools.MultiStatistics:
        """
        The :class:`deap.tools.MultiStatistics` objects to track fitness statistics.

        Uses the :attr:`FitnessFunction.db_titles` of each fitness of the 
        stage to set the chapter of the :class:`deap.tools.MultiStatistics` 
        object.
        If a name is set multiple times, a letter is appended to the name.
        For the fitness functions and global statistics the mean, max, min
        and std values are tracked for each generation.
        """
        if self._fitness_stats is None:
            capter_keys = []
            fitness_stats_dict = {}

            # Uses the index of the fitness to get the specific value that is stored
            # in the individual.
            def get_specific_fit_val(ind, index):
                return ind.fitness.values[index]

            fitness_names = [
                fitness_function.get_db_title() 
                for fitness_function in self.stage_data.fitness_functions
            ]
            if len(fitness_names) != len(set(fitness_names)):
                raise ValueError(
                    "Please set unique db_titles for the fitness functions."
                )

            for i, name in enumerate(fitness_names):
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

            self._fitness_stats = mstats

        return self._fitness_stats

    @property
    def fitness_logbook(self) -> tools.Logbook:
        if not hasattr(self, "_fitness_log"):
            self._fitness_log = tools.Logbook()

            stats_fields = []
            stats_fields = self.fitness_stats.fields
            self._fitness_log.header = ['nevals', 'gen'] + stats_fields # type: ignore

        return self._fitness_log

    @property
    def mutation_logbook(self) -> tools.Logbook:
        """A logbook to track how the mutations performed.

        The logbook uses the hash of the mutation as the chapter name.
        For each generation it tracks for each of the mutations the number of times
        it was called (key 'called') and how often it failed
        (key 'failed') to produce a valid individual.
        Additionally, it tracks if the mutated individual was selected in
        the survivor selection (key 'survivor').
        """
        if not hasattr(self, "_mutation_log"):
            self._mutation_log = tools.Logbook()

            stats_header = [
                mut.__hash__() for mut in self.stage_data.mutation_list
            ]
            self._mutation_log.header = ['gen'] + stats_header # type: ignore

            for stat in stats_header:
                self._mutation_log.chapters[stat].header = [ # type: ignore
                    'called', 'failed', 'survivor'
                ]

        return self._mutation_log

    @property
    def crossover_logbook(self) -> tools.Logbook:
        """A logbook to track how the crossovers performed.

        The logbook uses the hash of the crossover as the chapter name.
        For each generation and it tracks for each of the crossovers the number of times
        it was called (key 'called') and how often it failed
        (key 'failed') to produce a valid individual.
        Additionally, it tracks if the crossovered individual was selected in
        the survivor selection (key 'survivor').
        """
        if not hasattr(self, "_crossover_log"):
            self._crossover_log = tools.Logbook()

            stats_header = [
                cross.__hash__() for cross in self.stage_data.crossover_list
            ]
            self._crossover_log.header = ['gen'] + stats_header # type: ignore

            for stat in stats_header:
                self._crossover_log.chapters[stat].header = [ # type: ignore
                    'called', 'failed', 'survivor'
                ]

        return self._crossover_log

    def __perform_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual, bool, int]:
        """Selects a crossover and applies it to the parents.

        :return: The two offspring, a boolean indicating if the crossover was
            successful and the hash of the crossover to identify which one was
            used.
        """
        selected_crossover = random.choices(
            self.stage_data.crossover_list,
            weights=self.stage_data.crossover_weights
        )[0]

        return selected_crossover.crossover(parent1, parent2) + \
            (selected_crossover.__hash__(),)

    def __perform_mutation(
        self, individual: Individual
    ) -> tuple[Individual, bool, int]:
        """Selects a mutation and applies it to the individual.

        :return: The mutated individual, a boolean indicating if the mutation
            was successful and the hash of the mutation to identify which
            one was used.
        """

        selected_mutation = random.choices(
            self.stage_data.mutation_list,
            weights=self.stage_data.mutation_weights
        )[0]

        return selected_mutation.mutate(individual) + \
            (selected_mutation.__hash__(),)
    
    def __evaluate_individual(
        self, individual: Individual
    ) -> tuple[float, ...]:
        fitness_tuple = ()
        for fitness_function in self.stage_data.fitness_functions:
            fitness_tuple += (
                fitness_function.evaluate_individual(individual),
            )
        return fitness_tuple

    def __update_fitnesses(self,individuals: list[Individual]) -> int:
        """
        Checks which individuals in the population have invalid fitnesses.
        That means that the fitnesses have not been evaluated yet.
        Then evaluates the fitnesses of the invalid individuals.

        :return: The number of individuals with invalid fitnesses.
        """
        ic("Updating fitnesses.")
        invalid_ind = [ind for ind in individuals if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.reset() # Ensures that all features are reset
            ind.fitness.values = self.__evaluate_individual(ind)

        return len(invalid_ind)

    def __record_statistics(
        self,
        population: Population,
        nevals: int,
        gen: int,
        global_logbook: None | tools.Logbook = None,
        global_stats: None | tools.MultiStatistics = None,
    ) -> tuple[dict[str, dict[str, dict[str, int]]], dict[str, dict[str, dict[str, int]]] | None]:
        """
        Calculates the statistics of the population for the fitness and global 
        stats and stores them in the logbooks.

        :returns: A tuple with the fitness_record and the global_record.
        """
        fitness_record = self.fitness_stats.compile(population)
        self.fitness_logbook.record(gen=gen, nevals=nevals, **fitness_record)

        if global_stats is not None and global_logbook is not None:
            global_record = global_stats.compile(population)
            global_logbook.record(gen=gen, stage_id=self.id, **global_record)
        else:
            global_record = None

        return fitness_record, global_record

    def __create_offspring(self, population: Population) -> list[Individual]:
        """
        Creates offspring from the population by applying crossover and mutation.

        :return: The created offspring. Only the modified individuals are returned.
            These individuals have a .info attribute that stores the hash
            of the mutation and crossover that was used and a boolean 
            indicating if the mutation or crossover failed.

        """
        offspring = [base.deepcopy(ind) for ind in population]

        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            # Reset info about crossover, use None and True to have a consistent 
            # data type, so it can be checked if the crossover was used
            offspring[i - 1].info["cross_info"] = [None, True]
            offspring[i].info["cross_info"] = [None, True]

            # Perform crossover with given probability
            if random.random() < self.stage_data.crossover_probability:
                offspring[i - 1], offspring[i], success_bool, crossover_hash = self.__perform_crossover(
                    offspring[i - 1],
                    offspring[i]
                )

                # Reset fitness, features of the offsprings so they are recalculated
                offspring[i].reset()
                offspring[i - 1].reset()

                # Track which crossover was used and if it failed
                failed = not success_bool
                offspring[i - 1].info["cross_info"] = [crossover_hash, failed]
                offspring[i].info["cross_info"] = [crossover_hash, failed]


        for i in range(len(offspring)):
            # Reset info about mutation, use None and False to have a consistent 
            # data type, so it can be checked if the mutation was used
            offspring[i].info["mut_info"] = [None, True]

            # Perform mutation with given probability
            if random.random() < self.stage_data.mutation_probability:
                offspring[i], success_bool, mutation_hash = self.__perform_mutation(
                    offspring[i]
                )

                # Reset fitness, features of the mutated individual so they are recalculated
                offspring[i].reset()

                # Track if which mutation was used and if it failed
                failed = not success_bool
                offspring[i].info["mut_info"] = [mutation_hash, failed]

        # Pick out the individuals that were modified
        modified_offspring = []
        for i in range(len(offspring)):
            mut_info = offspring[i].info["mut_info"]
            cross_info = offspring[i].info["cross_info"]

            # Select only those, where the mutation or crossover did not fail
            # (also select those, where the mutation or crossover was not used)
            if mut_info[1] == False or cross_info[1] == False:
                modified_offspring.append(offspring[i])

        ic("Modified {} individuals".format(len(modified_offspring)))

        return modified_offspring

    def __track_successful_modifications(
        self,
        offspring: list[Individual],
        new_population: list[Individual],
    ) -> None:
        """Checks which of the offspring where selected as survivors.
        Adds entries to the 'cross_info' and 'mut_info' of the .info attribute
        of the offspring.
        The entries is a list with the hash of the mutation or crossover that 
        was used and a boolean indicating if the mutation or crossover failed 
        and finally if the individual was selected as a survivor.
        """
        for ind in offspring:
            # Check if the individual was selected as a survivor
            if ind in new_population:
                survivor = True
            else:
                survivor = False

            # Store the information in each individual to retrieve it later
            ind.info["cross_info"] = ind.info["cross_info"].append(survivor)
            ind.info["mut_info"] = ind.info["mut_info"].append(survivor)

    def __record_modification_log(
        self, 
        offspring: list[Individual], 
        gen: int, 
        modification_list: list[Mutation] | list[Crossover],
        info_key: str,
        modification_logbook: tools.Logbook,
    ) -> None:
        # Create the data structure to store the modification data
        # Use the hashes of all possible modifications as keys
        mod_data = { 
            str(modification.__hash__()): {
                "called": 0, 
                "failed": 0, 
                "survivor": 0
            } for modification in modification_list
        }
        for ind in offspring:

            # Load the data from the info attr of the individual
            hash = str(ind.info[info_key][0])
            mod_data[hash]["called"] += 1
            mod_data[hash]["failed"] += int(ind.info[info_key][1]) 
            mod_data[hash]["survivor"] += int(ind.info[info_key][2])

        modification_logbook.record(gen=gen, **mod_data)

    def __record_all_statistics_logs(
        self,
        population: Population,
        offspring: list[Individual],
        nevals: int,
        gen: int,
        global_stats: tools.MultiStatistics | None,
        global_log: tools.Logbook,
    ) -> None:

        # Save the data about the mutations and crossovers in the logbooks
        self.__record_modification_log(
            offspring=offspring,
            gen=gen,
            modification_list=self.stage_data.mutation_list,
            info_key="mut_info",
            modification_logbook=self.mutation_logbook,
        )
        self.__record_modification_log(
            offspring=offspring,
            gen=gen,
            modification_list=self.stage_data.crossover_list,
            info_key="cross_info",
            modification_logbook=self.crossover_logbook,
        )

        # Record the fitness statistics of the population
        fitness_record, global_record = self.__record_statistics(
            population=population,
            nevals=nevals,
            gen=gen,
            global_stats=global_stats,
            global_logbook=global_log,
        )

    def run(
        self, 
        population: Population,
        global_log: tools.Logbook,
        global_stats: tools.MultiStatistics | None,
    ) -> Population:
        # reset Individual and delete info attribute
        for ind in population:
            # Add fitness weights of current stage
            ind.fitness_weights = self.stage_data.fitness_weights

            ind.reset()
            ind.info = {}

        # Initialize the algorithm
        nevals = self.__update_fitnesses(population)
        # self.stage_data.hall_of_fame.update(population)
        fitness_record, global_record = self.__record_statistics(
            population=population,
            nevals=nevals,
            gen=0,
            global_stats=global_stats,
            global_logbook=global_log,
        )
        print(self.fitness_logbook.stream + "\t" + global_log.stream)

        # ── Run the evolution process ───────────────────────────────────────────
        for gen in range(1, self.stage_data.n_generations + 1):
            ic("Evolving Gen: ", gen)
            ic("Population size: ", len(population))

            parents = self.stage_data.parent_selection(population, len(population))
            ic("Selected {} parents".format(len(parents)))

            offspring = self.__create_offspring(parents)

            nevals = self.__update_fitnesses(offspring)

            # Combine the offspring with the population to select the survivors
            population_pool = population
            for ind in offspring:
                # Only add the offspring if it is not already in the population
                if ind not in population_pool:
                    population_pool.append(ind)
            ic("Created population pool")

            new_population = self.stage_data.survivor_selection(
                population_pool, len(population)
            )
            ic("Selected {} survivors".format(len(new_population)))

            # Check which offsprings were also selected as survivors
            self.__track_successful_modifications(
                offspring=offspring, new_population=new_population,
            )

            # Replace the old population with the new population
            population[:] = new_population

            # Track all data
            # self.stage_data.hall_of_fame.update(population)
            self.__record_all_statistics_logs(
                population=population,
                offspring=offspring,
                nevals=nevals,
                gen=gen,
                global_stats=global_stats,
                global_log=global_log,
            )
            print(self.fitness_logbook.stream + "\t" + global_log.stream)

            if self.stage_data.break_condition.check(population, gen):
                print("\nBreak condition was met. Stopping evolution.\n")
                break

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

