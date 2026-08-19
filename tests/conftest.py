import logging
import os
import tempfile
import time
import uuid
from importlib import import_module
from importlib.resources import files
from importlib.util import module_from_spec, spec_from_file_location
from typing import Callable

import numpy as np
import pytest
from ase.db.core import Database
from deap import tools

from fucrimodo.core import Individual, Population
from fucrimodo.core.abstracts import FitnessFunction, PopulationSelection, Stage
from fucrimodo.core.utils import CustomCellBounds, CustomClosestDistances
from fucrimodo.customs.ga_stage.crossovers import Crossover
from fucrimodo.customs.ga_stage.mutations import Mutation
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
    """A simple Individual with two periodic boundaries to use across multiple tests.

    (`You must promise me something. You will never go back to the surface, yes?`~Fujimoto)
    """
    return Individual(
        "H2",
        positions=[[0, 0, 5], [0, 0, 6]],
        cell=[3.0, 3.0, 15.0],  # vacuum along z
        pbc=(True, True, False),  # periodic in x,y only
    )


@pytest.fixture()
def closest_distances():
    return CustomClosestDistances(
        species=["H", "He", "Li", "Na", "Cl", "O"], ratio_of_covalent_radii=0.5
    )


@pytest.fixture()
def cell_bounds():
    return CustomCellBounds(
        bounds={
            "phi": [20, 160],
            "chi": [60, 120],
            "psi": [20, 160],
            "a": [1, 20],
            "b": [1, 20],
            "c": [1, 20],
        }
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


@pytest.fixture
def example_crossover(closest_distances):
    class ExampleCrossover(Crossover):
        """
        This Crossover just returns copies of the parents.
        """

        def _perform_crossover(
            self, parent1: Individual, parent2: Individual
        ) -> tuple[Individual, Individual] | tuple[None, None]:
            return (parent1.copy(), parent2.copy())

    return ExampleCrossover(closest_distances)


@pytest.fixture
def example_mutation(closest_distances):
    class ExampleMutation(Mutation):
        """
        This Mutation just returns copies of the parent.
        """

        def _perform_mutation(self, individual: Individual) -> Individual | None:
            return individual.copy()

    return ExampleMutation(closest_distances)


@pytest.fixture
def example_selection():
    class ExampleSelection(PopulationSelection):
        def select(self, individuals: list[Individual], n: int) -> list[Individual]:
            return individuals[:n]

    return ExampleSelection()


@pytest.fixture
def mutation_reproducability_assessment():
    def assess(individual: Individual, mut: Mutation, assess_change=True):

        # Save initial state of rng
        original_state = mut._rng.bit_generator.state

        # run multiple tests
        gen_inds = []
        successes = []
        for _ in range(5):
            ind, success = mut.mutate(individual.copy())
            gen_inds.append(ind)
            successes.append(success)
            assert all(ind.pbc == individual.pbc)
            if success:
                assert ind != individual

        # Check if there was at least one success
        assert any(successes)

        # Check reproducability
        # Restore original big generator state
        mut._rng.bit_generator.state = original_state
        for i in range(5):
            ind, success = mut.mutate(individual.copy())
            assert ind == gen_inds[i]
            assert success == successes[i]

        if assess_change:
            # Check that results can change
            res_changed = []
            for i in range(5):
                ind, _ = mut.mutate(individual.copy())
                res_changed.append(ind != gen_inds[i])
            assert any(res_changed)

    return assess


@pytest.fixture(scope="session")
def run_data_path(tmp_path_factory):
    """Generate run data once and reuse it for all tests.

    If this is used in a test please mark the test as slow with
    @pytest.mark.slow

    """
    save_dir = tmp_path_factory.mktemp("save_dir")

    atoms_path = (
        files("fucrimodo") / "lab_template" / "data" / "raw" / "test-target.xyz"
    )

    # Create target_file
    target_file_path = os.path.join(save_dir, "test-target-file.json")
    create_target_file_data = import_module(
        "fucrimodo.lab_template.configs.utils.create_target_file_data"
    )
    create_target_file_data.main(
        atoms_path=atoms_path,
        save_path=target_file_path,
        verbose=False,
    )

    module = import_module("fucrimodo.lab_template.configs.run.test_run_config")
    module.main(
        name="test_run",
        save_dir=save_dir,
        target_file_path=target_file_path,
        n_parallel=4,
        verbose=False,
    )

    return os.path.join(save_dir, "test_run")
