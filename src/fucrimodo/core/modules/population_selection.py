from abc import ABC, abstractmethod
from .individual import Individual

# ╔══════════════════════════════════════════════════════════╗
# ║         Abstract Base Class for StartPopulation          ║
# ╚══════════════════════════════════════════════════════════╝


class PopulationSelection(ABC):
    """Class that defines the abstract base class for population selection.

    Population selection is used to select individuals from the population
    based on a selection strategy.
    This is often used in the genetic algorithms to select individuals that
    get modified or used for the next generation.
    """

    def __init__(self):
        pass

    @abstractmethod
    def select(self, individuals: list[Individual], n: int) -> list[Individual]:
        """Method that selects individuals from a given list of individuals.

        The selection strategy must be implemented in this method.
        NOTE: Here we use a list of individuals to avoid that population
        is overwritten, since it stores data of the run.

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
