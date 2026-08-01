from collections.abc import Sequence
from fucrimodo.core.modules import Population, Individual, FitnessFunction
import numpy as np


def _seperate_fitness_and_weights(
    fitness_functions: (
        Sequence[FitnessFunction | tuple[FitnessFunction, float]] | FitnessFunction
    ),
) -> tuple[list[FitnessFunction], tuple[float, ...]]:
    """Returns the fitness functions seperatly from weights

    If no weights are given for a fitness, a weight of 1.0 is assumed.
    """
    # If single fitness with weight is given raise error
    # since it doesnt make sence
    assert (
        not type(fitness_functions) is tuple
    ), "A single fitness function cannot have a weight, please do not assign it."

    # If only one fitness is given just return it with its weight
    if isinstance(fitness_functions, FitnessFunction):
        return [fitness_functions], (1.0,)

    # Seperate the fitness functions and weights
    fitness_func_list = []
    weights = ()
    for fit_weight_tuple in fitness_functions:
        # If tuple is given seperate the function and weight
        if isinstance(fit_weight_tuple, tuple):
            fitness_func_list.append(fit_weight_tuple[0])
            weights += (fit_weight_tuple[1],)

        # If only function is given assign weight as 1.
        else:
            fitness_func_list.append(fit_weight_tuple)
            weights += (1.0,)

    return fitness_func_list, weights


def assign_fitness_to_individual(
    individual: Individual,
    fitness_functions: (
        Sequence[FitnessFunction | tuple[FitnessFunction, float]] | FitnessFunction
    ),
    force_reset_storage: bool = False,
) -> None:
    """Evaluates and assigns the fitness of individual for each fitness function.

    It automatically resets the fitness storage of the individual, if weights do not
    match the previous weights. Reset of fitness storage can also be forced with
    :param:`force_reset_storage`.
    """
    fitness_func_list, weights = _seperate_fitness_and_weights(fitness_functions)

    if force_reset_storage or (weights != individual.fitness.weights):
        individual.set_new_fitness_storage(weights)

    individual.fitness.values = [
        f.evaluate_individual(individual) for f in fitness_func_list
    ]


def assign_fitness_to_individuals(
    individuals: list[Individual],
    fitness_functions: (
        Sequence[FitnessFunction | tuple[FitnessFunction, float]] | FitnessFunction
    ),
    force_reset_storage: bool = False,
) -> None:
    """Evaluates and assigns the fitness of individuals for each fitness function.

    Can speed up the evaluation by evaluating all individuals at once.

    It automatically resets the fitness storage of the individuals, if weights do not
    match the previous weights. Reset of fitness storage can also be forced with
    :param:`force_reset_storage`.
    """
    fitness_func_list, weights = _seperate_fitness_and_weights(fitness_functions)

    # Evaluate the fitnesses of the individuals for each fitness function
    fitness_matrix = []
    for fitness_func in fitness_func_list:
        fitnesses = fitness_func.evaluate_individuals(individuals)
        fitness_matrix.append(fitnesses)

    # Get fitness tuple for each individual and assign it
    for ind, ind_fitness in zip(individuals, np.array(fitness_matrix).T):
        if force_reset_storage or (weights != ind.fitness.weights):
            ind.set_new_fitness_storage(weights)
        ind.fitness.values = tuple(ind_fitness)


def assign_fitness_to_population(
    population: Population,
    fitness_functions: (
        Sequence[FitnessFunction | tuple[FitnessFunction, float]] | FitnessFunction
    ),
    force_reset_storage: bool = False,
) -> None:
    """Evaluates and Assigns fitness to all individuals in the population.

    It automatically resets the fitness storage of the individuals, if weights do not
    match the previous weights. Reset of fitness storage can also be forced with
    :param:`force_reset_storage`.
    """
    assign_fitness_to_individuals(
        population.individuals, fitness_functions, force_reset_storage
    )
