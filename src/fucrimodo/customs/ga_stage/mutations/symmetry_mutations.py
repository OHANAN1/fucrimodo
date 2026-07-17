# TODO: Fix weird error
import matid

from fucrimodo.core.modules import Individual
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances

from .abstract import Mutation


class GetConventionalCellMutation(Mutation):
    """Tries to change the individual to its conventional cell.

    This mutation is based on the `matid` :class:`SymmetryAnalyzer` class.
    It will try to convert the individual to its conventional cell by
    performing a symmetry analysis.
    The cell size and atomic positions will be changed accordingly.

    :param closest_distances: The closest distances object.
    :param symmetry_tol: The symmetry tolerance to be used in the symmetry
        analysis.
    :param max_volume_increase: The maximum volume increase allowed when
        converting the cell to the conventional cell.
    :param max_volume_decrease: The maximum volume decrease allowed when
        converting the cell to the conventional cell.
    :param max_steps: The maximum number of times the mutation should be
        retried when it failed.
        Not recommended to increase this value above 1 since the mutation
        is deterministic and will return the same result.

    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        symmetry_tol: float | None = 1e-5,
        max_volume_increase: float = 1.2,
        max_volume_decrease: float = 0.8,
        max_steps: int = 1,
    ):
        self.max_steps = max_steps
        self.symmetry_tol = symmetry_tol
        self.closest_distances = closest_distances
        self.max_volume_increase = max_volume_increase
        self.max_volume_decrease = max_volume_decrease

    def perform_mutation(self, individual: Individual) -> Individual | None:
        # Get the original volume
        original_volume = individual.get_volume()

        individual.set_pbc([True, True, True])

        # Analyze the individual to get the conventional cell
        analyzer = matid.SymmetryAnalyzer(individual.copy(), self.symmetry_tol)
        conventional_cell = analyzer.get_conventional_system()

        # Create a new individual with the analyzed properties
        positions = conventional_cell.get_positions()
        atomic_numbers = conventional_cell.get_atomic_numbers()
        cell = conventional_cell.get_cell()
        offspring = Individual(atomic_numbers, positions=positions, cell=cell)

        # Check if the volume increase is too large
        if offspring.get_volume() > original_volume * self.max_volume_increase:
            return None

        # Check if the volume decrease is too large
        if offspring.get_volume() < original_volume * self.max_volume_decrease:
            return None

        # Check if the new individual is the same as the original
        # If so, return None to symbolize that the mutation was not successful
        if offspring == individual:
            return None

        return offspring
