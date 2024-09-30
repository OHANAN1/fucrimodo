from abc import ABC, abstractmethod
import ase

from .individual import Individual

# ╔══════════════════════════════════════════════════════════╗
# ║         Abstract Base Class for StartPopulation          ║
# ╚══════════════════════════════════════════════════════════╝

class PopulationSelection(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def select_start_pop(
        self,
        individuals: list[Individual]
    ) -> list[Individual]:
        """
        Returns parts of the given crystals based on the selection strategy
        of the class.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
