import copy
import functools
from fucrimodo.core.modules import Population, FitnessFunction, Individual
from fucrimodo.core.modules.population_selection import PopulationSelection
from .mutations import Mutation
from .crossovers import Crossover
from .break_conditions import BreakCondition
from typing import Sequence
from deap import tools
import numpy as np
import random

import logging
# logger = logging.getLogger('run_logger')

class GeneticAlgorithm:
    def __init__(
        self,
        fitness_functions: Sequence[FitnessFunction],
        fitness_weights: Sequence[float],
        crossover_list: Sequence[Crossover],
        crossover_weights: Sequence[float],
        mutation_list: Sequence[Mutation],
        mutation_weights: Sequence[float],
        mutation_probability: float,
        crossover_probability: float,
        break_condition: BreakCondition,
        parent_selection: PopulationSelection,
        parent_ratio: float,
        survivor_selection: PopulationSelection,
        save_n_best_crystals: int = 10,
    ):
        self.fitness_functions = fitness_functions
        self.fitness_weights = fitness_weights
        self.crossover_list = crossover_list
        self.crossover_weights = crossover_weights
        self.mutation_list = mutation_list
        self.mutation_weights = mutation_weights
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability
        self.break_condition = break_condition
        self.parent_selection = parent_selection
        self.survivor_selection = survivor_selection
        self._hall_of_fame = tools.HallOfFame(save_n_best_crystals)
        self.parent_ratio = parent_ratio

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, "_logger"):
            raise AttributeError("No logger set. Please set a logger.")
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

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
        if not hasattr(self, "_fitness_stats"):
            capter_keys = []
            fitness_stats_dict = {}

            # Uses the index of the fitness to get the specific value that is stored
            # in the individual.
            def get_specific_fit_val(ind, index):
                return ind.fitness.values[index]

            fitness_names = [
                fitness_function.db_title
                for fitness_function in self.fitness_functions
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
                str(mut.__hash__()) for mut in self.mutation_list
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
                str(cross.__hash__()) for cross in self.crossover_list
            ]
            self._crossover_log.header = ['gen'] + stats_header # type: ignore

            for stat in stats_header:
                self._crossover_log.chapters[stat].header = [ # type: ignore
                    'called', 'failed', 'survivor'
                ]

        return self._crossover_log

    @property
    def hall_of_fame(self) -> tools.HallOfFame:
        return self._hall_of_fame

    @property
    def generation(self) -> int:
        """The current generation of the genetic algorithm."""
        # If not set, initialize the generation with 0
        if not hasattr(self, "_generation"):
            self._generation = 0
        return self._generation

    def __perform_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual, bool, int]:
        """Selects a crossover and applies it to the parents.

        :return: The two offspring, a boolean indicating if the crossover was
            successful and the hash of the crossover to identify which one was
            used.
        """
        selected_crossover = random.choices(
            self.crossover_list,
            weights=self.crossover_weights
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
            self.mutation_list, weights=self.mutation_weights
        )[0]

        return selected_mutation.mutate(individual) + \
            (selected_mutation.__hash__(),)
    
    def __evaluate_individual(
        self, individual: Individual
    ) -> tuple[float, ...]:
        fitness_tuple = ()
        for fitness_function in self.fitness_functions:
            fitness_tuple += (
                fitness_function.evaluate_individual(individual),
            )
        return fitness_tuple

    def __evaluate_individuals(
        self, individuals: list[Individual]
    ) -> list[tuple[float, ...]]:
        """Evaluates the fitnesses of the individuals for each fitness function.

        Speeds up the evaluation by evaluating all fitnesses of an individual
        at once.
        """
        # Create a list of empty tuples for each individual
        fitness_tuples_list: list[tuple[float, ...]] = [
            () for _ in range(len(individuals))
        ]

        # Evaluate the fitnesses of the individuals for each fitness function
        for fitness_function in self.fitness_functions:
            # Use the fitness function to evaluate the fitnesses of the individuals
            fitnesses = fitness_function.evaluate_individuals(individuals)

            # Append the fitnesses to the corresponding tuple of each individual
            for ind_index in range(len(fitness_tuples_list)):
                fitness_tuples_list[ind_index] += (fitnesses[ind_index],)

        return fitness_tuples_list

    def __update_fitnesses(self,individuals: list[Individual]) -> int:
        """
        Checks which individuals in the population have invalid fitnesses.
        That means that the fitnesses have not been evaluated yet.
        Then evaluates the fitnesses of the invalid individuals.

        :return: The number of individuals with invalid fitnesses.
        """
        invalid_ind = [ind for ind in individuals if not ind.fitness.valid]

        if len(invalid_ind) == 0:
            return 0

        for ind in invalid_ind:
            ind.reset() # Ensures that all features are reset

        # Evaluate the fitnesses of the invalid individuals
        fitness_tuples_list = self.__evaluate_individuals(invalid_ind)

        # Assign the fitnesses to the individuals
        for ind, fitness_tuple in zip(invalid_ind, fitness_tuples_list):
            ind.fitness.values = fitness_tuple

        return len(invalid_ind)

    def __record_statistics(
        self,
        population: Population,
        nevals: int,
        gen: int,
        stage_id: int,
        global_stats: tools.MultiStatistics | None,
        global_log: tools.Logbook,
    ) -> tuple[dict[str, dict[str, dict[str, int]]], dict[str, dict[str, dict[str, int]]] | None]:
        """
        Calculates the statistics of the population for the fitness and global 
        stats and stores them in the logbooks.
        Also updates the hall of fame with the best individuals of the population.

        :returns: A tuple with the fitness_record and the global_record.
        """
        fitness_record = self.fitness_stats.compile(population.individuals)
        self.fitness_logbook.record(gen=gen, nevals=nevals, **fitness_record)

        if global_stats is not None and global_log is not None:
            global_record = global_stats.compile(population.individuals)
            # Record the global statistics. Use the stage_id to identify the stage
            # and the generation of the population to track the total generation
            # and not only the generation of the stage.
            global_log.record(
                gen=population.generation, stage_id=stage_id, **global_record
            )
        else:
            global_record = None

        self.hall_of_fame.update(population.individuals)

        return fitness_record, global_record

    def __create_offspring(
        self, 
        individuals: list[Individual]
    ) -> list[Individual]:
        """
        Creates offspring from the population by applying crossover and mutation.

        :return: The created offspring. Only the modified individuals are returned.
            These individuals have a .info attribute that stores the hash
            of the mutation and crossover that was used and a boolean 
            indicating if the mutation or crossover failed.

        """
        offspring = copy.deepcopy(individuals)

        # Apply crossover and mutation on the offspring
        for i in range(1, len(offspring), 2):
            # Reset info about crossover, use None and True to have a consistent 
            # data type, so it can be checked if the crossover was used
            offspring[i - 1].info["cross_info"] = [None, True]
            offspring[i].info["cross_info"] = [None, True]

            # Perform crossover with given probability
            if random.random() < self.crossover_probability:
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
        
        # For uneven populations, the last individual is not crossed
        # Assign the .info['cross_info'] attribute to the last individual
        if len(offspring) % 2 == 1:
            offspring[-1].info["cross_info"] = [None, True]

        for i in range(len(offspring)):
            # Reset info about mutation, use None and False to have a consistent 
            # data type, so it can be checked if the mutation was used
            offspring[i].info["mut_info"] = [None, True]

            # Perform mutation with given probability
            if random.random() < self.mutation_probability:
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

        self.logger.info("Modified {} individuals".format(len(modified_offspring)))

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
            ind.info["cross_info"].append(survivor)
            ind.info["mut_info"].append(survivor)

    def __record_modification_log(
        self, 
        offspring: list[Individual], 
        gen: int, 
        modification_list: Sequence[Mutation] | Sequence[Crossover],
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

            # Skip if the hash is None or "None". Happens if no mutation or
            # crossover was to modify the individual due to the probability.
            if hash is None or hash == "None":
                continue

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
        stage_id: int,
        global_stats: tools.MultiStatistics | None,
        global_log: tools.Logbook,
    ) -> None:

        # Save the data about the mutations and crossovers in the logbooks
        self.__record_modification_log(
            offspring=offspring,
            gen=gen,
            modification_list=self.mutation_list,
            info_key="mut_info",
            modification_logbook=self.mutation_logbook,
        )
        self.__record_modification_log(
            offspring=offspring,
            gen=gen,
            modification_list=self.crossover_list,
            info_key="cross_info",
            modification_logbook=self.crossover_logbook,
        )

        # Record the fitness statistics of the population
        self.__record_statistics(
            population=population,
            nevals=nevals,
            gen=gen,
            stage_id=stage_id,
            global_stats=global_stats,
            global_log=global_log,
        )

    def __initialize_evolution(
        self,
        population: Population,
        stage_id: int,
        global_stats: tools.MultiStatistics | None,
        global_log: tools.Logbook,
    ) -> None:
        # Resets the and deletes the info attribute of each individual.
        # Adds the fitness weights of the current stage to each individual.
        for ind in population.individuals:
            ind.fitness_weights = self.fitness_weights
            ind.reset()
            ind.info = {}

        nevals = self.__update_fitnesses(population.individuals)

        self.__record_statistics(
            population=population,
            nevals=nevals,
            gen=0,
            stage_id=stage_id,
            global_stats=global_stats,
            global_log=global_log
        )

    def __attach_logger_to_mut_and_cross(self, logger: logging.Logger) -> None:
        """Attaches the logger to mutations and crossovers of stage.

        This method is used to attach the logger to all objects of the stage
        that have a logger attribute. This is useful to have a consistent
        logging behavior in the stage.
        """
        self.logger.info("Attaching logger to the mutations and crossovers of the GA.")
        for obj in [
            *self.crossover_list,
            *self.mutation_list,
        ]:
            obj.logger = logger

    def run(
        self,
        population: Population,
        stage_id: int,
        global_stats: tools.MultiStatistics | None,
        global_log: tools.Logbook,
    ) -> Population:
        # Store the initial population size
        population_size = population.size

        # Attach the logger to all objects of the stage
        self.__attach_logger_to_mut_and_cross(self.logger)

        # Initialize the evolution process
        self.__initialize_evolution(
            population=population, 
            global_stats=global_stats,
            global_log=global_log,
            stage_id=stage_id
        )

        print(global_log.stream)

        self._generation = 0
        while not self.break_condition.check(
            population.individuals, self.generation
        ):
            self._generation += 1

            # ── Run the evolution process ────────────────────────────────────
            self.logger.info(f"Evolving Gen: {self._generation}")
            self.logger.info(f"Population size: {population.size}")

            # ── Select Parents ───────────────────────────────────────────────
            parents = self.parent_selection.select(
                population.individuals, int(population_size * self.parent_ratio)
            )
            self.logger.info("Selected {} parents".format(len(parents)))

            # ── Create Offspring ─────────────────────────────────────────────
            offspring = self.__create_offspring(parents)
            nevals = self.__update_fitnesses(offspring)

            # Combine the offspring with the population to select the survivors
            population_pool = population.individuals 

            # Only add the offspring that are not already in the population
            for ind in offspring:
                if ind not in population_pool:
                    population_pool.append(ind)
            self.logger.info("Created population pool")

            # Select survivors from the old population and offspring
            new_population = self.survivor_selection.select(
                population_pool, population_size
            )
            self.logger.info("Selected {} survivors".format(len(new_population)))

            # Check which offsprings were also selected as survivors
            self.__track_successful_modifications(
                offspring=offspring, new_population=new_population,
            )

            # Replace the old population with the new population
            population.individuals = new_population

            # Track all data
            self.__record_all_statistics_logs(
                population=population,
                offspring=offspring,
                nevals=nevals,
                gen=self._generation,
                global_stats=global_stats,
                global_log=global_log,
                stage_id=stage_id,
            )
            print(global_log.stream, end="\r")

        return population
