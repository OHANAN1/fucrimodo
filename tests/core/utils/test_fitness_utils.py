import pytest
from fucrimodo.core.utils import fitness_utils


def test_seperate_fitness_and_weights(example_fitness):
    # Test that it works with single fitness
    fit_funcs, weights = fitness_utils._seperate_fitness_and_weights(example_fitness)
    assert fit_funcs == [example_fitness]
    assert weights == (1.0,)

    # Test that single fitness cannot have a weight
    with pytest.raises(AssertionError):
        fitness_utils._seperate_fitness_and_weights((example_fitness, 0.1))  # type: ignore

    # Test that list of fitnesses with weights is seperated
    fit_funcs, weights = fitness_utils._seperate_fitness_and_weights(
        [example_fitness, (example_fitness, 0.5)]
    )
    assert fit_funcs == [example_fitness, example_fitness]
    assert weights == (1.0, 0.5)


def test_assign_fitness_to_individual(example_fitness, ind_slab):
    fitness_utils.assign_fitness_to_individual(
        individual=ind_slab,
        fitness_functions=[example_fitness, (example_fitness, 0.5)],
    )

    assert ind_slab.fitness.values == (2, 2)
    assert ind_slab.fitness_weights == (1.0, 0.5)


def test_assign_fitness_to_individuals(
    example_fitness, ind_molecule, ind_slab, ind_crystal
):
    individuals = [ind_molecule, ind_slab, ind_crystal]

    fitness_utils.assign_fitness_to_individuals(
        individuals=individuals,
        fitness_functions=[example_fitness, (example_fitness, 0.5)],
    )

    # Fitness value here is (abitrarily) the number of periodic boundary conditions
    assert ind_molecule.fitness.values == (0, 0)
    assert ind_slab.fitness.values == (2, 2)
    assert ind_crystal.fitness.values == (3, 3)

    # Check that weights are assigned
    assert ind_molecule.fitness_weights == (1.0, 0.5)
    assert ind_crystal.fitness_weights == (1.0, 0.5)
    assert ind_slab.fitness_weights == (1.0, 0.5)


def test_assign_fitness_to_population(example_fitness, population):
    fitness_utils.assign_fitness_to_population(
        population=population,
        fitness_functions=[example_fitness, (example_fitness, 0.5)],
    )

    # Fitness value here is (abitrarily) the number of periodic boundary conditions
    assert population.individuals[0].fitness.values == (2, 2)
    assert population.individuals[1].fitness.values == (3, 3)
    assert population.individuals[2].fitness.values == (0, 0)

    # Check that weights are assigned
    assert population.individuals[0].fitness_weights == (1.0, 0.5)
    assert population.individuals[1].fitness_weights == (1.0, 0.5)
    assert population.individuals[2].fitness_weights == (1.0, 0.5)
