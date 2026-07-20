import numpy as np
import pytest

from fucrimodo.core.modules import Individual, Population


@pytest.fixture
def ind_molecule():
    """A simple molecular Individual to use across multiple tests."""
    return Individual("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])


@pytest.fixture
def ind_crystal():
    """A simple Individual with periodic boundaries to use across multiple tests."""
    return Individual(
        "NaCl",
        positions=[[0, 0, 0], [2.82, 2.82, 2.82]],
        cell=[5.64, 5.64, 5.64],
        pbc=True,
    )


@pytest.fixture
def ind_slab():
    """A simple Individual with two periodic boundaries to use across multiple tests."""
    return Individual(
        "H2",
        positions=[[0, 0, 5], [0, 0, 6]],
        cell=[3.0, 3.0, 15.0],  # vacuum along z
        pbc=(True, True, False),  # periodic in x,y only
    )


@pytest.fixture
def population(ind_slab, ind_crystal, ind_molecule):
    return Population([ind_slab, ind_crystal, ind_molecule])


@pytest.fixture
def scored_individual(ind):
    """An Individual with weights and a valid fitness already set."""
    ind.fitness_weights = (1.0, -1.0)
    ind.fitness.values = (2.0, 3.0)
    return ind


@pytest.fixture
def rng():
    """Deterministic random generator for reproducible tests."""
    return np.random.default_rng(seed=42)
