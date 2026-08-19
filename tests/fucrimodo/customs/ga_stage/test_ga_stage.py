import numpy as np
import pytest
from deap import tools

from fucrimodo.customs.break_conditions import GenerationBreak
from fucrimodo.customs.fitness_functions import PhysicalityFitness
from fucrimodo.customs.ga_stage.ga_stage import GAStage
from fucrimodo.customs.ga_stage.genetic_algorithm import GeneticAlgorithm


@pytest.fixture
def ga_stage(
    example_fitness,
    example_crossover,
    example_mutation,
    example_selection,
    closest_distances,
):
    return GAStage(
        name="Test_Stage",
        fitness_functions=[
            example_fitness,
            (PhysicalityFitness(closest_distances=closest_distances), 0.5),
        ],
        crossover_list=[example_crossover, (example_crossover, 0.5)],
        mutation_list=[example_mutation, (example_mutation, 0.5)],
        mutation_probability=0.5,
        crossover_probability=0.5,
        break_condition=GenerationBreak(generation_limit=10),
        parent_selection=example_selection,
        survivor_selection=example_selection,
        parent_ratio=0.5,
        description="Test Description",
        save_n_structures=10,
    )


def test___init__(ga_stage):
    assert hasattr(ga_stage, "_cfg")
    assert len(ga_stage._cfg) > 0
    assert type(ga_stage._cfg) is dict

    assert hasattr(ga_stage, "ga_runner")
    assert type(ga_stage.ga_runner) is GeneticAlgorithm

    assert ga_stage.name == "Test_Stage"
    assert ga_stage.description == "Test Description"


def test__seperate_object_weight_tuple(ga_stage: GAStage, example_fitness):
    objs, weights = ga_stage._seperate_object_weight_tuples([("a", 1.0), "b", ("c", 3)])

    assert objs == ["a", "b", "c"]
    assert weights == (1.0, 1.0, 3)


def test__save_hall_of_fame(ga_stage):
    pass


def test__save_crossovers(ga_stage):
    pass


def test__save_mutations(ga_stage):
    pass


def test__save_fitnesses(ga_stage):
    pass


def test_info_dict(ga_stage: GAStage):
    assert "break_condition" in ga_stage.info_dict
    assert "parent_selection" in ga_stage.info_dict
    assert "n_generations" in ga_stage.info_dict


def test_run(ga_stage: GAStage, population, logger):

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

    # Init stage
    ga_stage.id = 1
    ga_stage.logger = logger

    # Run the GA Stage
    new_pop = ga_stage.run(population, global_log=global_log, global_stats=mstats)

    assert len(global_log) == 10

    # TODO: Add more testing here
