import pytest

from fucrimodo.core import Individual
from fucrimodo.core.abstracts import FitnessFunction


class MockFitnessFunction(FitnessFunction):
    def __init__(self, var1, var2, db_title="MockFitnessFunction"):
        super().__init__(db_title=db_title)
        self.var1 = var1
        self.var2 = var2

    def evaluate_individual(self, individual: Individual) -> float:
        return 1.0


def test_subclass_without_evaluate_cannot_instantiate():
    class Incomplete(FitnessFunction):
        pass  # does not implement evaluate_individual

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore


@pytest.mark.parametrize(
    "var1, var2, expected",
    [
        (0, 1, "MockFitnessFunction(var1=0, var2=1)"),
        ([-1, 0, 1], "String", "MockFitnessFunction(var1=[-1, 0, 1], var2=String)"),
        (None, True, "MockFitnessFunction(var1=None, var2=True)"),
    ],
)
def test_repr(var1, var2, expected):
    break_cond = MockFitnessFunction(var1, var2)
    assert repr(break_cond) == expected


def test_mock_evaluate(ind_crystal, ind_molecule, ind_slab):
    break_cond = MockFitnessFunction(None, None)

    assert break_cond.evaluate_individual(ind_crystal) == 1.0
    assert break_cond.evaluate_individuals([ind_crystal, ind_molecule, ind_slab]) == [
        1.0,
        1.0,
        1.0,
    ]
