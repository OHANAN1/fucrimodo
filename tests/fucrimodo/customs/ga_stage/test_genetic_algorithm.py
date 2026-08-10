from fucrimodo.core.abstracts.fitness_function import FitnessFunction
from fucrimodo.core.population import Population
import pytest
import numpy as np
import copy

from fucrimodo.customs.ga_stage.genetic_algorithm import GeneticAlgorithm, norm_weights
from fucrimodo.customs.ga_stage.mutations import pos_mut
from fucrimodo.customs.ga_stage.crossovers import (
    OnePointElementCrossover,
    OnePointPositionCrossover,
)
from fucrimodo.customs.population_selections import TournamentSelection
from fucrimodo.customs.break_conditions import GenerationBreak
from fucrimodo.customs.fitness_functions import PhysicalityFitness
from deap import tools
import pandas as pd


def test_norm_weights():
    # With list
    weights = norm_weights([1.0, 0.5, 2.0])
    assert weights.sum() == pytest.approx(1)

    # With tuple
    weights = norm_weights((1.0, 0.7, 7.0))
    assert weights.sum() == pytest.approx(1)


def perform_run(genetic_algorithm, population):
    # Generate the statistics dicts for data tracking
    # This stat is non-sence but for testing reproducability
    stats_dict = {
        "positions_sum": tools.Statistics(
            key=lambda ind: np.sum(ind.positions.flatten())
        )
    }
    mstats = tools.MultiStatistics(**stats_dict)
    mstats.register("avg", np.mean)
    mstats.register("max", np.max)
    mstats.register("min", np.min)
    mstats.register("std", np.std)

    global_log = tools.Logbook()
    global_stats_fields = mstats.fields
    global_log.header = ["stage_id", "gen"] + global_stats_fields  # type: ignore

    new_pop = genetic_algorithm.run(
        population,
        stage_id=1,
        global_stats=mstats,
        global_log=global_log,
    )
    return new_pop, global_log


class TestGeneticAlgorithm:
    @pytest.fixture
    def genetic_algorithm(
        self,
        example_fitness,
        closest_distances,
    ):
        rng = np.random.default_rng(42)

        mutations = [
            pos_mut.RattleMutation(
                closest_distances=closest_distances,
                rng=rng,
                rattle_strength=0.01,
                rattle_prop=1.0,
            ),
            pos_mut.RattleMutation(
                closest_distances=closest_distances,
                rng=rng,
                rattle_strength=0.1,
                rattle_prop=1.0,
            ),
        ]

        crossovers = [
            OnePointElementCrossover(closest_distances=closest_distances, rng=rng),
            OnePointPositionCrossover(closest_distances=closest_distances, rng=rng),
        ]

        return GeneticAlgorithm(
            fitness_functions=[
                example_fitness,
                PhysicalityFitness(closest_distances=closest_distances),
            ],
            fitness_weights=(1.0, 0.5),
            crossover_list=crossovers,
            crossover_weights=(1.0, 0.5),
            mutation_list=mutations,
            mutation_weights=(1.0, 0.5),
            mutation_probability=0.9,
            crossover_probability=0.9,
            break_condition=GenerationBreak(generation_limit=10),
            parent_selection=TournamentSelection(tournament_size=1, rng=rng),
            parent_ratio=0.5,
            survivor_selection=TournamentSelection(tournament_size=1, rng=rng),
            save_n_best_individuals=10,
            rng=rng,
        )

    def test___init__(self, genetic_algorithm: GeneticAlgorithm):
        # Mut and Cross weights get normalized
        assert sum(genetic_algorithm.norm_crossover_weights) == pytest.approx(1.0)
        assert sum(genetic_algorithm.norm_mutation_weights) == pytest.approx(1.0)

        assert hasattr(genetic_algorithm, "_hall_of_fame")
        assert genetic_algorithm._hall_of_fame.maxsize == 10

    def test_run(
        self,
        genetic_algorithm: GeneticAlgorithm,
        ind_crystal,
        ind_slab,
        ind_molecule,
        logger,
    ):
        initial_gen = 5
        original_pop = Population([ind_crystal, ind_slab, ind_molecule])
        original_pop.generation = initial_gen

        genetic_algorithm.logger = logger

        new_pop, logbook = perform_run(
            genetic_algorithm,
            population=original_pop.copy(),
        )
        assert len(new_pop) == 3
        assert new_pop.generation == 10 + initial_gen
        assert len(logbook) == 10
        assert len(genetic_algorithm._crossover_log) == 10
        assert len(genetic_algorithm._mutation_log) == 10
        assert (
            len(genetic_algorithm._fitness_log)
            == 10 + 1  # since initial pop is considered
        )

        logbook_df = pd.DataFrame(logbook.chapters["positions_sum"])
        assert min(logbook_df["gen"]) == 1 + initial_gen
        assert max(logbook_df["gen"]) == 10 + initial_gen

        cross_hash = str(genetic_algorithm.crossover_list[0].__hash__())
        assert (
            min(
                pd.DataFrame(genetic_algorithm._crossover_log.chapters[cross_hash])[
                    "gen"
                ]
            )
            == 1
        )
        assert (
            max(
                pd.DataFrame(genetic_algorithm._crossover_log.chapters[cross_hash])[
                    "gen"
                ]
            )
            == 10
        )

        assert (
            min(
                pd.DataFrame(
                    genetic_algorithm._fitness_log.chapters["ExampleFitnessFunction"]
                )["gen"]
            )
            == 0
        )
        assert (
            max(
                pd.DataFrame(
                    genetic_algorithm._fitness_log.chapters["ExampleFitnessFunction"]
                )["gen"]
            )
            == 10
        )

    def test_reproducability(
        self,
        ind_crystal,
        ind_slab,
        ind_molecule,
        genetic_algorithm,
    ):
        original_state = genetic_algorithm._rng.bit_generator.state

        original_pop = Population(
            [
                ind_crystal,
                ind_slab,
                ind_molecule,
                ind_crystal.copy(),
                ind_slab.copy(),
                ind_molecule.copy(),
            ]
        )
        original_pop.generation = 0

        first_pop, first_logbook = perform_run(
            genetic_algorithm, population=original_pop.copy()
        )
        first_log_df = pd.DataFrame(first_logbook.chapters["positions_sum"])

        # Reset everything to initial state
        genetic_algorithm._rng.bit_generator.state = original_state

        pop_copy = original_pop.copy()
        pop_copy.generation = 0

        second_pop, second_logbook = perform_run(genetic_algorithm, population=pop_copy)
        second_log_df = pd.DataFrame(second_logbook.chapters["positions_sum"])

        assert all(first_log_df == second_log_df)
        assert first_pop.individuals == second_pop.individuals
        assert first_pop.generation == second_pop.generation

        # -------- Test if change
        # Reset only population but not the rng state
        pop_copy = original_pop.copy()
        pop_copy.generation = 0

        third_pop, third_logbook = perform_run(genetic_algorithm, population=pop_copy)
        third_log_df = pd.DataFrame(third_logbook.chapters["positions_sum"])

        assert any(third_log_df != second_log_df)
        assert third_pop != second_pop
