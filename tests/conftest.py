import time
import pytest
import logging
import tempfile
import uuid
import os
from deap import tools
from ase.db.core import Database
from typing import Callable
import numpy as np

from fucrimodo.core.abstracts import Stage, FitnessFunction
from fucrimodo.core import Individual, Population
from fucrimodo.customs.global_soap_target import GlobalSOAP


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
def periodic_soap_obj():
    return GlobalSOAP(
        r_cut=2,
        n_max=2,
        l_max=2,
        species=["H", "Na", "Cl", "O"],
        periodic=True,
        average="outer",
    )


@pytest.fixture
def unperiodic_soap_obj():
    return GlobalSOAP(
        r_cut=2,
        n_max=2,
        l_max=2,
        species=["H", "Na", "Cl", "O"],
        periodic=False,
        average="outer",
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
def example_fitness():
    class ExampleFitnessFunction(FitnessFunction):
        def evaluate_individual(self, individual: Individual) -> float:
            return np.sum(individual.pbc)

    return ExampleFitnessFunction()


@pytest.fixture
def rng():
    """Deterministic random generator for reproducible tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def logger():
    return logging.getLogger()


@pytest.fixture
def temp_dir():
    return os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))


@pytest.fixture
def ExampleStage():
    class ExampleStage(Stage):
        def run(
            self,
            population: Population,
            global_log: tools.Logbook,
            global_stats: tools.MultiStatistics | None,
        ) -> Population:
            if global_stats:
                global_record = global_stats.compile(population.individuals)
                global_log.record(
                    gen=population.generation, stage_id=self.id, **global_record
                )

            time.sleep(0.05)
            self.was_run = True
            return population

        def save_results(
            self,
            save_dir: str,
            structures_db: Database,
            global_statistics_dict: (
                dict[str, Callable[[Individual], float]] | None
            ) = None,
        ) -> None:
            self.was_saved = True
            return None

        @property
        def info_dict(self) -> dict:
            return {}

    return ExampleStage
