from abc import ABC, abstractmethod
from .population import Population


class BreakCondition(ABC):
    """Checks if algorithm should stop based on the state of the population.

    Children of this class should implement the :meth:`check` method, to test if
    the algorithm should be stoped based on the state of the population or any
    other algorithm attribute (e.g. generation number).
    """

    @abstractmethod
    def check(self, population: Population, info: dict | None = None) -> bool:
        """Method to check if the break condition is fullfilled.

        :params population: Population for which the break condition should be checked.
        :params info: Used to pass additional that are needed to check the condition.
            (e.g. generation number)

        :returns: True if break condition is fulfilled and False if not.
        """
        pass

    def __repr__(self):
        class_name = self.__class__.__name__
        variables = vars(self)
        variables_str = ", ".join(f"{key}={value}" for key, value in variables.items())
        return f"{class_name}({variables_str})"
