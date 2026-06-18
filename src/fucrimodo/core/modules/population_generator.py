from abc import ABC, abstractmethod

from fucrimodo.core.modules.individual import Individual
from .population import Population

class PopulationGenerator(ABC):
    def __init__(self) -> None:
        pass

    def generate_population(self, size: int) -> Population:
        """Method to generate a population of a given size.

        :param size: The size of the population that should be generated.

        :return: The generated population.
        """
        return Population(self.generate_individuals(n=size))

    @abstractmethod
    def generate_individuals(self, n: int) -> list[Individual]:
        """Method to generate a list of individuals that are used to create a 
        population.

        :param n: The number of individuals that should be generated.

        :return: The generated individual.
        """
        pass
