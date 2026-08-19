import pytest

from fucrimodo.core.utils.fitness_utils import assign_fitness_to_population
from fucrimodo.customs.break_conditions import (
    GenerationBreak,
    MaxFitnessBreak,
    MinFitnessBreak,
    MultipleAndBreak,
    MultipleOrBreak,
    NeverBreak,
    NotBreak,
)


class TestNeverBreak:
    def test_check(self, population):
        assert not NeverBreak().check(population=population, info={"example": 5})


class TestMaxFitnessBreak:
    def test_check(self, population, example_fitness):
        assign_fitness_to_population(population, [example_fitness, example_fitness])

        # Test not break, i.e. fitness is below threshold
        assert not MaxFitnessBreak(fitness_index=0, fitness_threshold=3.1).check(
            population
        )

        # Test break, i.e. fitness is above threshold
        assert MaxFitnessBreak(fitness_index=0, fitness_threshold=-1.0).check(
            population
        )


class TestMinFitnessBreak:
    def test_check(self, population, example_fitness):
        assign_fitness_to_population(population, [example_fitness, example_fitness])

        # Test not break, i.e. fitness is below threshold
        assert MinFitnessBreak(fitness_index=0, fitness_threshold=3.1).check(population)

        # Test break, i.e. fitness is above threshold
        assert not MinFitnessBreak(fitness_index=0, fitness_threshold=-1.0).check(
            population
        )


class TestNotBreak:
    def test_check(self, population):
        # Inverts the results
        assert NotBreak(NeverBreak()).check(population)


class TestMultipleAndBreak:
    def test_check(self, population, example_fitness):
        assign_fitness_to_population(population, [example_fitness, example_fitness])

        assert MultipleAndBreak(
            [MaxFitnessBreak(fitness_index=0, fitness_threshold=-1.0), NeverBreak()]
        ).check(population) is (True and False)
        assert MultipleAndBreak(
            [
                MaxFitnessBreak(fitness_index=0, fitness_threshold=-1.0),
                MaxFitnessBreak(fitness_index=0, fitness_threshold=-1.0),
            ]
        ).check(population) is (True and True)
        assert MultipleAndBreak([NeverBreak(), NeverBreak()]).check(population) is (
            False and False
        )


class TestMultipleOrBreak:
    def test_check(self, population, example_fitness):
        assign_fitness_to_population(population, [example_fitness, example_fitness])

        assert MultipleOrBreak(
            [MaxFitnessBreak(fitness_index=0, fitness_threshold=-1.0), NeverBreak()]
        ).check(population) is (True or False)
        assert MultipleOrBreak(
            [
                MaxFitnessBreak(fitness_index=0, fitness_threshold=-1.0),
                MaxFitnessBreak(fitness_index=0, fitness_threshold=-1.0),
            ]
        ).check(population) is (True or True)
        assert MultipleOrBreak([NeverBreak(), NeverBreak()]).check(population) is (
            False or False
        )


class TestGenerationBreak:
    def test_check(self, population, example_fitness):
        assign_fitness_to_population(population, [example_fitness, example_fitness])

        # Raises error if no generation attr is passed
        with pytest.raises(AssertionError):
            GenerationBreak(generation_limit=5).check(population)

        # Raises error if generation is not a number
        with pytest.raises(AssertionError):
            GenerationBreak(generation_limit=5).check(
                population, info={"generation_index": "value"}
            )

        # Test not breaking
        assert not GenerationBreak(generation_limit=5).check(
            population, info={"generation_index": 4}
        )

        # Test breaking
        assert GenerationBreak(generation_limit=5).check(
            population, info={"generation_index": 5}
        )
