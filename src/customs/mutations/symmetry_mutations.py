from matid import SymmetryAnalyzer
from .abstract_mutation import Mutation
from ...utils.closest_distances_class import CustomClosestDistances
import ase


class GetConventionalCellMutation(Mutation):
    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        symmetry_tol: float | None = 1e-5,
    ):
        self.max_steps = 1
        self.symmetry_tol = symmetry_tol
        self.closest_distances = closest_distances

    def perform_mutation(self, crystal: ase.Atoms) -> ase.Atoms | None:
        analyzer = SymmetryAnalyzer(crystal.copy(), self.symmetry_tol)
        conventional_cell = analyzer.get_conventional_system()

        positions = conventional_cell.get_positions()
        atomic_numbers = conventional_cell.get_atomic_numbers()
        cell = conventional_cell.get_cell()

        offspring = ase.Atoms(
            atomic_numbers,
            positions=positions,
            cell=cell
        )

        return offspring
