from abc import ABC, abstractmethod


class BreakCondition(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def check(self, population: list, generation_index: int | None = None) -> bool:
        pass

    def __repr__(self):
        class_name = self.__class__.__name__
        variables = vars(self)
        variables_str = ", ".join(f"{key}={value}" for key, value in variables.items())
        return f"{class_name}({variables_str})"


class NeverBreak(BreakCondition):
    def check(self, population: list, generation_index: int | None = None) -> bool:
        return False

    def __repr__(self):
        return "NeverBreak()"


class MaxFitnessBreak(BreakCondition):

    def __init__(self, fitness_index: int, fitness_threshold: float):
        self.fitness_threshold = fitness_threshold
        self.fitness_index = fitness_index

    def check(self, population: list, generation_index: int | None = None) -> bool:
        fitness_values = [
            individual.fitness.values[self.fitness_index] for individual in population
        ]
        return max(fitness_values) >= self.fitness_threshold

    def __repr__(self):
        return f"f[{self.fitness_index}] >= {self.fitness_threshold}"


class MinFitnessBreak(BreakCondition):
    def __init__(self, fitness_index: int, fitness_threshold: float):
        self.fitness_threshold = fitness_threshold
        self.fitness_index = fitness_index

    def check(self, population: list, generation_index: int | None = None) -> bool:
        fitness_values = [
            individual.fitness.values[self.fitness_index] for individual in population
        ]
        return min(fitness_values) <= self.fitness_threshold

    def __repr__(self):
        return f"f[{self.fitness_index}] <= {self.fitness_threshold}"


class MultipleAndBreak(BreakCondition):
    def __init__(self, break_conditions: list):
        self.break_conditions = break_conditions

    def check(self, population: list, generation_index: int | None = None) -> bool:
        return all(
            [
                break_condition.check(population, generation_index)
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
    def __init__(self, break_conditions: list):
        self.break_conditions = break_conditions

    def check(self, population: list, generation_index: int | None = None) -> bool:
        return any(
            [
                break_condition.check(population, generation_index)
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
    def __init__(self, break_condition: BreakCondition):
        self.break_condition = break_condition

    def check(self, population: list, generation_index: int | None = None) -> bool:
        return not self.break_condition.check(population, generation_index)

    def __repr__(self):
        return f"not {self.break_condition}"


class GenerationBreak(BreakCondition):

    def __init__(self, generation_limit: int):
        self.generation_limit = generation_limit

    def check(self, population: list, generation_index: int | None = None) -> bool:
        assert (
            generation_index is not None
        ), "Please provide generation index for this break condition to work."
        return generation_index >= self.generation_limit

    def __repr__(self):
        return f"gen >= {self.generation_limit}"
