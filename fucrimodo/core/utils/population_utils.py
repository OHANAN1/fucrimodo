from collections.abc import Sequence
from fucrimodo.core.modules import Population, Individual, FitnessFunction

def evaluate_individuals(
    individuals: list[Individual],
    fitness_functions: Sequence[FitnessFunction | tuple[FitnessFunction, float]],
) -> list[tuple[float, ...]]:
    """Evaluates the fitnesses of the individuals for each fitness function.

    Speeds up the evaluation by evaluating all fitnesses of an individual
    at once.
    Be aware that this function resets the individuals as well as their info
    attribute.
    It also assigns the weights to the individuals each time it is called.
    """
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
            weights += (1.,)

    # Resets the and deletes the info attribute of each individual.
    # Also assigns the weights to the individuals.
    for ind in individuals:
        ind.fitness_weights = weights
        ind.reset()
        ind.info = {}

    # Create a list of empty tuples for each individual
    fitness_tuples_list: list[tuple[float, ...]] = [
        () for _ in range(len(individuals))
    ]

    # Evaluate the fitnesses of the individuals for each fitness function
    for fitness_func in fitness_func_list:
        # Use the fitness function to evaluate the fitnesses of the individuals
        fitnesses = fitness_func.evaluate_individuals(individuals)

        # Append the fitnesses to the corresponding tuple of each individual
        for ind_index in range(len(fitness_tuples_list)):
            fitness_tuples_list[ind_index] += (fitnesses[ind_index],)

    return fitness_tuples_list


def assign_fitness_to_all_individuals(
    population: Population,
    fitness_functions: Sequence[FitnessFunction | tuple[FitnessFunction, float]],
) -> None:
    """Assigns fitness with fitness function to all individuals in the population.
    """
    # Evaluate the fitnesses of the invalid individuals
    fitness_tuples_list = evaluate_individuals(population.individuals,
                                               fitness_functions)

    # Assign the fitnesses to the individuals
    for ind, fitness_tuple in zip(population.individuals, fitness_tuples_list):
        ind.fitness.values = fitness_tuple
