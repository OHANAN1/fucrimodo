import pytest
from fucrimodo.core.modules import Population


@pytest.fixture
def population(ind_slab, ind_crystal, ind_molecule):
    return Population([ind_slab, ind_crystal, ind_molecule])


def test_population_individuals(population, ind_slab, ind_crystal, ind_molecule):
    individuals = [ind_slab, ind_crystal, ind_molecule]

    # Test if individuals are correctly set
    for i, ind in enumerate(population.individuals):
        assert ind == individuals[i]


def test_size(population):
    assert len(population) == 3
    assert population.size == 3


def test_generation(population):
    # Test start generation
    assert population.generation == 0

    # Test manually setting generation
    population.generation = 1
    assert population.generation == 1

    # Test automatic generation update if new inds are set
    old_inds = population.individuals
    population.individuals = old_inds
    assert population.generation == 2
