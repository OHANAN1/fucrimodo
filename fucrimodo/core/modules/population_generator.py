from abc import ABC, abstractmethod
from .population import Population

class PopulationGenerator(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate_population(self, size: int) -> Population:
        """Method to generate a population of a given size.

        :param size: The size of the population that should be generated.

        :return: The generated population.
        """
        pass
