from abc import ABC, abstractmethod

from fucrimodo.core.modules.individual import Individual
from .population import Population


class PopulationGenerator(ABC):
    """Abstract Base class for generating a population.

    The population generator should create the individuals of a population.
    """

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

        :return: A list of the generated individual.
        """
        pass
