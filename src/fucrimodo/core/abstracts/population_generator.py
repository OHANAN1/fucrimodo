from abc import ABC, abstractmethod

from ..individual import Individual
from ..population import Population


class PopulationGenerator(ABC):
    """Generates a new population.

    The population generator should create the individuals of a population with
    the :meth:`generate_individuals` method.
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
