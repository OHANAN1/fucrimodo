import pytest
from fucrimodo.core.modules import BreakCondition, Population


class MockBreakCondition(BreakCondition):
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2

    def check(self, population: Population, info: dict | None = None) -> bool:
        if info:
            return True
        else:
            return False


def test_subclass_without_check_cannot_instantiate():
    class Incomplete(BreakCondition):
        pass  # does not implement check

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore


@pytest.mark.parametrize(
    "var1, var2, expected",
    [
        (0, 1, "MockBreakCondition(var1=0, var2=1)"),
        ([-1, 0, 1], "String", "MockBreakCondition(var1=[-1, 0, 1], var2=String)"),
        (None, True, "MockBreakCondition(var1=None, var2=True)"),
    ],
)
def test_repr(var1, var2, expected):
    break_cond = MockBreakCondition(var1, var2)
    assert repr(break_cond) == expected


def test_mock_check(population):
    break_cond = MockBreakCondition(None, None)

    assert break_cond.check(population) == False
    assert break_cond.check(population, info={}) == False
    assert break_cond.check(population, info={"key": "val"}) == True
