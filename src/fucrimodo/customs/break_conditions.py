from fucrimodo.core.abstracts import BreakCondition
from fucrimodo.core import Population


class NeverBreak(BreakCondition):
    """Never break the algorithm.

    (`You can't be busy - you're five!` ~ Kôichi)
    """

    def check(self, population: Population, info: dict | None = None) -> bool:
        return False

    def __repr__(self):
        return "NeverBreak()"


class MultipleAndBreak(BreakCondition):
    """Break condition that triggers when all provided break conditions are fulfilled.

    :param break_conditions: List of break conditions to evaluate with a logical AND.
    """

    def __init__(self, break_conditions: list):
        self.break_conditions = break_conditions

    def check(self, population: Population, info: dict | None = None) -> bool:
        return all(
            break_condition.check(population, info)
            for break_condition in self.break_conditions
        )

    def __repr__(self):
        return (
            "( "
            + " and ".join(
                break_condition.__repr__() for break_condition in self.break_conditions
            )
            + " )"
        )


class MultipleOrBreak(BreakCondition):
    """Break condition that triggers when any of the provided conditions triggers.

    :param break_conditions: List of break conditions to evaluate with a logical OR.
    """

    def __init__(self, break_conditions: list):
        self.break_conditions = break_conditions

    def check(self, population: Population, info: dict | None = None) -> bool:
        return any(
            break_condition.check(population, info)
            for break_condition in self.break_conditions
        )

    def __repr__(self):
        return (
            "( "
            + " or ".join(
                break_condition.__repr__() for break_condition in self.break_conditions
            )
            + " )"
        )


class NotBreak(BreakCondition):
    """Break condition that negates another break condition.

    :param break_condition: Break condition whose result should be inverted.
    :type break_condition: BreakCondition
    """

    def __init__(self, break_condition: BreakCondition):
        self.break_condition = break_condition

    def check(self, population: Population, info: dict | None = None) -> bool:
        return not self.break_condition.check(population, info)

    def __repr__(self):
        return f"not {self.break_condition}"


class MaxFitnessBreak(BreakCondition):
    """Break condition that triggers when the maximum fitness value is at or above a threshold.

    The maximum is computed over all individuals for the fitness value at
    ``fitness_index``.

    :param fitness_index: Index of the fitness value to inspect in each individual.
    :type fitness_index: int
    :param fitness_threshold: Threshold the maximum fitness must reach or exceed.
    :type fitness_threshold: float
    """

    def __init__(self, fitness_index: int, fitness_threshold: float):
        self.fitness_threshold = fitness_threshold
        self.fitness_index = fitness_index

    def check(self, population: Population, info: dict | None = None) -> bool:
        fitness_values = [
            individual.fitness.values[self.fitness_index]
            for individual in population.individuals
        ]
        return max(fitness_values) >= self.fitness_threshold

    def __repr__(self):
        return f"f[{self.fitness_index}] >= {self.fitness_threshold}"


class MinFitnessBreak(BreakCondition):
    """Break condition that triggers when the minimum fitness value is at or below a threshold.

    The minimum is computed over all individuals for the fitness value at
    ``fitness_index``.

    :param fitness_index: Index of the fitness value to inspect in each individual.
    :param fitness_threshold: Threshold the minimum fitness must not fall below.
    """

    def __init__(self, fitness_index: int, fitness_threshold: float):
        self.fitness_threshold = fitness_threshold
        self.fitness_index = fitness_index

    def check(self, population: Population, info: dict | None = None) -> bool:
        fitness_values = [
            individual.fitness.values[self.fitness_index]
            for individual in population.individuals
        ]
        return min(fitness_values) <= self.fitness_threshold

    def __repr__(self):
        return f"f[{self.fitness_index}] <= {self.fitness_threshold}"


class GenerationBreak(BreakCondition):
    """Break condition that triggers when the generation index reaches the limit.

    .. note::

        The caller must supply the current generation index via ``info["generation_index"]``
        when calling :meth:`check`.

    :param generation_limit: Maximum generation index allowed before the condition triggers.
    """

    def __init__(self, generation_limit: int):
        self.generation_limit = generation_limit

    def check(self, population: Population, info: dict | None = None) -> bool:
        assert (
            info
            and ("generation_index" in info.keys())
            and isinstance(info["generation_index"], (int, float))
        ), "Please provide generation_index, through the info dict for this break condition to work."

        return info["generation_index"] >= self.generation_limit

    def __repr__(self):
        return f"gen >= {self.generation_limit}"
