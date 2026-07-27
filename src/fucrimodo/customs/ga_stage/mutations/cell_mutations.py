import random

import ase_ga.standardmutations as ase_standard_mut
import numpy as np
from ase import build
from ase.cell import Cell

from fucrimodo.core.modules.individual import Individual
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances

from .abstract import Mutation


class ScaleUnitCellMutation(Mutation):
    """
    Scales the unit cell by a random factor.
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        cell_bounds: CustomCellBounds,
        max_scale: float = 2.0,
        min_scale: float = 0.5,
        scale_atoms: bool = True,
        max_steps: int = 100,
        n_variable_cell_vectors: int = 3,
    ):
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.scale_atoms = scale_atoms
        self.max_steps = max_steps
        self.closest_distances = closest_distances
        self.cell_bounds = cell_bounds
        self.n_variable_cell_vectors = n_variable_cell_vectors

    def perform_mutation(self, individual: Individual) -> Individual | None:
        offspring = individual.copy()
        cell = offspring.get_cell()[:]  # type: ignore

        random_factor = np.random.uniform(self.min_scale, self.max_scale)

        # Select random cell vectors to scale
        cell_indicees = np.random.choice(
            [0, 1, 2], self.n_variable_cell_vectors, replace=False
        )
        for i in cell_indicees:
            cell[i] *= random_factor

        offspring.set_cell(
            Cell(cell), scale_atoms=self.scale_atoms, apply_constraint=True
        )

        # If structure did not change, return None, to avoid false positives
        if offspring == individual:
            self.logger.debug("Structure did not change.")
            return None

        # Check if the new cell is within the bounds
        if not self.cell_bounds.is_within_bounds(offspring.cell):
            self.logger.debug("Structure is outside the bounds.")
            return None

        return offspring


class StrainMutation(Mutation):
    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        n_variable_cell_vectors: int = 3,
        cell_bounds: CustomCellBounds | None = None,
        stddev: float = 0.7,
    ) -> None:
        self.closest_distances = closest_distances
        self.n_variable_cell_vectors = n_variable_cell_vectors
        self.cell_bounds = cell_bounds
        self.max_steps = 1
        self.stddev = stddev

    def perform_mutation(self, individual: Individual) -> Individual | None:
        if self.cell_bounds is None:
            len_ang = individual.cell.cellpar()
            a = len_ang[0]
            b = len_ang[1]
            c = len_ang[2]

            cell_bounds = CustomCellBounds(
                {
                    "a": [a - 1.0, a + 1.0],
                    "b": [b - 1.0, b + 1.0],
                    "c": [c - 1.0, c + 1.0],
                }
            )
        else:
            cell_bounds = self.cell_bounds

        ase_strain = ase_standard_mut.StrainMutation(
            blmin=self.closest_distances,
            number_of_variable_cell_vectors=self.n_variable_cell_vectors,
            cellbounds=cell_bounds,
            stddev=self.stddev,
            verbose=True,
        )
        ase_strain.update_scaling_volume([individual])

        offspring = individual
        mutant = ase_strain.mutate(offspring)
        return mutant


class EnlargeMutation(Mutation):
    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        cell_bounds: CustomCellBounds,
        max_steps: int = 1,
    ) -> None:
        self.closest_distances = closest_distances
        self.max_steps = max_steps
        self.cell_bounds = cell_bounds

    def __get_possible_new_cell(
        self, cell_vectors: np.ndarray
    ) -> tuple[Cell, list[int]] | tuple[None, None]:
        possible_sides = []
        possible_cell_vectors = []
        for i in range(3):
            cell_vectors_copy = cell_vectors.copy()
            cell_vectors_copy[i] *= 2
            test_cell = Cell(cell_vectors_copy)

            if self.cell_bounds.is_within_bounds(test_cell):
                possible_sides.append(i)
                possible_cell_vectors.append(cell_vectors_copy[i])
            else:
                possible_cell_vectors.append(cell_vectors[i])

        if len(possible_sides) == 0:
            return None, None

        else:
            return Cell(possible_cell_vectors), possible_sides

    def perform_mutation(self, individual: Individual) -> Individual | None:
        offspring = individual
        cell = offspring.get_cell()
        cell_vectors = cell[:]  # type: ignore

        possible_cell, possible_sides = self.__get_possible_new_cell(cell_vectors)

        if possible_cell is None:
            return None
        else:
            new_cell = possible_cell

            repeat_sequence = [1, 1, 1]
            for i in possible_sides:  # type: ignore
                repeat_sequence[i] += 1

            offspring = offspring.repeat(repeat_sequence)
            offspring.set_cell(new_cell, scale_atoms=False, apply_constraint=False)

            return offspring


class NiggliReduceMutation(Mutation):
    def __init__(
        self, closest_distances: CustomClosestDistances, max_steps: int = 1
    ) -> None:
        self.closest_distances = closest_distances
        self.max_steps = max_steps

    def perform_mutation(self, individual: Individual) -> Individual | None:
        build.niggli_reduce(individual)
        return individual


class MinimizeTiltMutation(Mutation):
    def __init__(
        self, closest_distances: CustomClosestDistances, max_steps: int = 1
    ) -> None:
        self.closest_distances = closest_distances
        self.max_steps = max_steps

    def perform_mutation(self, individual: Individual) -> Individual | None:
        build.minimize_tilt(individual)
        return individual


class CutoutMutation(Mutation):
    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        cell_bounds: CustomCellBounds,
        tolerance: float = 0.01,
        max_steps: int = 50,
    ) -> None:
        self.tolerance = tolerance
        self.closest_distances = closest_distances
        self.cell_bounds = cell_bounds
        self.max_steps = max_steps

    def perform_mutation(self, individual: Individual) -> Individual | None:
        if not self.cell_bounds.is_within_bounds(individual.cell):
            return None

        a_vec = (np.random.uniform(0.0, 0.8), 0, 0)
        b_vec = (0, np.random.uniform(0.0, 0.8), 0)
        if random.choice([True, False]):
            c_vec = (0, 0, np.random.uniform(0.0, 0.8))
        else:
            c_vec = None

        cutout_individual = build.cut(
            individual,
            a=a_vec,
            b=b_vec,
            c=c_vec,
            clength=None,
            tolerance=self.tolerance,
        )

        if self.cell_bounds.is_within_bounds(cutout_individual.cell):
            if len(cutout_individual) <= 1:
                return None
            else:
                return cutout_individual
        else:
            return None


class RotationMutation(Mutation):
    def __init__(
        self, closest_distances: CustomClosestDistances, max_steps: int = 30
    ) -> None:
        self.closest_distances = closest_distances
        self.max_steps = max_steps

    def perform_mutation(self, individual: Individual) -> Individual | None:
        v_rand = np.random.choice(["x", "y", "z"])
        a_rand = np.random.uniform(0, 90)
        individual.rotate(a=a_rand, v=v_rand, rotate_cell=False)
        return individual
