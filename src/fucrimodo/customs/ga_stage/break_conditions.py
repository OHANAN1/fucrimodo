from fucrimodo.core.modules import BreakCondition
from fucrimodo.core.modules.population import Population


class NeverBreak(BreakCondition):
    """Never break the algorithm."""

    def check(self, population: Population, info: dict | None = None) -> bool:
        return False

    def __repr__(self):
        return "NeverBreak()"


class MultipleAndBreak(BreakCondition):
    """Check if the condition of all of the provided `break_conditions` is fullfilled."""

    def __init__(self, break_conditions: list):
        self.break_conditions = break_conditions

    def check(self, population: Population, info: dict | None = None) -> bool:
        return all(
            [
                break_condition.check(population, info)
                for break_condition in self.break_conditions
            ]
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
    """Check if the condition of either one of the provided `break_conditions` is fullfilled."""

    def __init__(self, break_conditions: list):
        self.break_conditions = break_conditions

    def check(self, population: Population, info: dict | None = None) -> bool:
        return any(
            [
                break_condition.check(population, info)
                for break_condition in self.break_conditions
            ]
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
    """Check if the condition of the provided `break_conditions` is not fullfilled."""

    def __init__(self, break_condition: BreakCondition):
        self.break_condition = break_condition

    def check(self, population: Population, info: dict | None = None) -> bool:
        return not self.break_condition.check(population, info)

    def __repr__(self):
        return f"not {self.break_condition}"


class MaxFitnessBreak(BreakCondition):
    """Break the algorithm if the fitness at index `fitness_index` of one of the individuals is above the `fitness_threshold`."""

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
    """Break the algorithm if the fitness at index `fitness_index` of one of the individuals is below the `fitness_threshold`."""

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
    """Checks if the current generation is above the generation_limit.

    Please provide the current generation index with the key 'generation_index'
    in the `info` dictionary of the `check` method.
    """

    def __init__(self, generation_limit: int):
        self.generation_limit = generation_limit

    def check(self, population: Population, info: dict | None = None) -> bool:
        """Method to check if the current generation is above the generation_limit.

        :params info: Please provide the current generation index with the key 'generation_index'.
        """
        assert (
            info
            and ("generation_index" in info.keys())
            and (type(info["generation_index"]) is (int or float))
        ), "Please provide generation_index, through the info dict for this break condition to work."

        return info["generation_index"] >= self.generation_limit

    def __repr__(self):
        return f"gen >= {self.generation_limit}"
