import pytest

from fucrimodo.core import Individual, Population
from fucrimodo.core.abstracts import PopulationGenerator


class MockPopulationGenerator(PopulationGenerator):
    def __init__(self, individual):
        self.individual = individual

    def generate_individuals(self, n: int) -> list[Individual]:
        return [self.individual for _ in range(n)]


def test_subclass_without_generate_cannot_instantiate():
    class Incomplete(PopulationGenerator):
        pass  # does not implement evaluate_individual

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore


def test_mock_generate(ind_crystal):
    pop_generator = MockPopulationGenerator(ind_crystal)

    assert pop_generator.generate_individuals(n=1) == [ind_crystal]
    assert pop_generator.generate_individuals(n=2) == [ind_crystal, ind_crystal]

    pop = pop_generator.generate_population(size=3)
    assert pop.individuals == [ind_crystal, ind_crystal, ind_crystal]
    assert pop.size == 3
    assert isinstance(pop, Population)
