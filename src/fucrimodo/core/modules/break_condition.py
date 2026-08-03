from abc import ABC, abstractmethod
from .population import Population


class BreakCondition(ABC):
    """Check if algorithm should stop based on the state of the population.

    This should implement a check based on the state of the population
    """

    @abstractmethod
    def check(self, population: Population, info: dict | None = None) -> bool:
        """Method to check if the break condition is fullfilled.

        :params population: Population for which the break condition should be checked.
        :params info: Used to pass additional that are needed to check the condition.

        :returns: True if break condition is fulfilled and False if not.
        """
        pass

    def __repr__(self):
        class_name = self.__class__.__name__
        variables = vars(self)
        variables_str = ", ".join(f"{key}={value}" for key, value in variables.items())
        return f"{class_name}({variables_str})"
