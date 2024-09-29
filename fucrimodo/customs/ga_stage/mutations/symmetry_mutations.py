# TODO: Fix weird error
# from matid import SymmetryAnalyzer
from fucrimodo.core.modules import Individual
from .abstract import Mutation
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
import ase


class GetConventionalCellMutation(Mutation):
    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        symmetry_tol: float | None = 1e-5,
    ):
        raise NotImplementedError
        self.max_steps = 1
        self.symmetry_tol = symmetry_tol
        self.closest_distances = closest_distances

    def perform_mutation(self, crystal: Individual) -> Individual | None:
        raise NotImplementedError
        # analyzer = SymmetryAnalyzer(crystal.copy(), self.symmetry_tol)
        # conventional_cell = analyzer.get_conventional_system()
        #
        # positions = conventional_cell.get_positions()
        # atomic_numbers = conventional_cell.get_atomic_numbers()
        # cell = conventional_cell.get_cell()
        #
        # offspring = Individual(
        #     atomic_numbers,
        #     positions=positions,
        #     cell=cell
        # )
        #
        # return offspring
