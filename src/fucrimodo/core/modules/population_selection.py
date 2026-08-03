from abc import ABC, abstractmethod
from .individual import Individual

# ╔══════════════════════════════════════════════════════════╗
# ║         Abstract Base Class for StartPopulation          ║
# ╚══════════════════════════════════════════════════════════╝


class PopulationSelection(ABC):
    """Select a subset of the population of individuals.

    Population selection is used to select individuals from the population based
    on a selection strategy. This is often used in the genetic algorithms to
    select individuals that get modified or used for the next generation.
    """

    @abstractmethod
    def select(self, individuals: list[Individual], n: int) -> list[Individual]:
        """Method that selects individuals from a given list of individuals.

        The selection strategy must be implemented in this method.
        NOTE: Contrary to the name the method does not use the population but a list of
        individuals. This is often the desired chase, since the selected individuals
        often do not directly build the new population. To use individuals of a population
        do the following:

        .. code-block:: python

            selected_individuals = population_selection.select(population.individuals)


        :param individuals: A list of individuals that are used for
            the selection.
        :param n: The number of individuals that should be selected.

        :return: A list of individuals that were selected based on the
            implemented selection strategy.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
