import ase_ga.standardmutations as ase_standard_mut
import numpy as np
from ase import build
from ase.cell import Cell

from ....core import Individual
from ....core.utils import CustomCellBounds, CustomClosestDistances

from .abstract import Mutation


class ScaleUnitCellMutation(Mutation):
    """
    Mutate an individual by scaling its unit cell vectors.

    A single random scaling factor is drawn uniformly from
    ``[min_scale, max_scale]`` and applied to
    ``n_variable_cell_vectors`` randomly selected cell vectors. Atomic
    positions are scaled together with the cell when ``scale_atoms`` is
    ``True``.

    (`Ponyo, make the candle bigger!` ~ Sosuke)

    :param closest_distances: Minimum allowed interatomic distances used
        to validate the mutated structure.
    :type closest_distances: CustomClosestDistances
    :param cell_bounds: Allowed bounds for the mutated cell.
    :type cell_bounds: CustomCellBounds
    :param max_scale: Upper bound of the uniform scaling factor
        distribution. Defaults to ``2.0``.
    :type max_scale: float
    :param min_scale: Lower bound of the uniform scaling factor
        distribution. Defaults to ``0.5``.
    :type min_scale: float
    :param scale_atoms: Whether to scale atomic coordinates together with
        the cell vectors. Defaults to ``True``.
    :type scale_atoms: bool
    :param n_variable_cell_vectors: Number of cell vectors to scale,
        chosen without replacement from ``[0, 1, 2]``. Defaults to ``3``.
    :type n_variable_cell_vectors: int
    :param max_retries: Maximum number of attempts to produce a valid
        mutation. Defaults to ``100``.
    :type max_retries: int
    :param rng: Random number generator. If ``None``, the base class
        creates one.
    :type rng: None | np.random.Generator
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        cell_bounds: CustomCellBounds,
        max_scale: float = 2.0,
        min_scale: float = 0.5,
        scale_atoms: bool = True,
        n_variable_cell_vectors: int = 3,
        max_retries: int = 100,
        rng: None | np.random.Generator = None,
    ):
        super().__init__(
            closest_distances=closest_distances, max_retries=max_retries, rng=rng
        )
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.scale_atoms = scale_atoms
        self.cell_bounds = cell_bounds
        self.n_variable_cell_vectors = n_variable_cell_vectors

    def _perform_mutation(self, individual: Individual) -> Individual | None:
        offspring = individual.copy()
        cell = offspring.get_cell()[:]  # type: ignore

        random_factor = self._rng.uniform(self.min_scale, self.max_scale)

        # Select random cell vectors to scale
        cell_indicees = self._rng.choice(
            [0, 1, 2], self.n_variable_cell_vectors, replace=False
        )
        for i in cell_indicees:
            cell[i] *= random_factor

        offspring.set_cell(
            Cell(cell), scale_atoms=self.scale_atoms, apply_constraint=True
        )

        # If structure did not change, return None, to avoid false positives
        if offspring == individual:
            if self.logger:
                self.logger.debug("Structure did not change.")
            return None

        # Check if the new cell is within the bounds
        if not self.cell_bounds.is_within_bounds(offspring.cell):
            if self.logger:
                self.logger.debug("Structure is outside the bounds.")
            return None

        return offspring


class StrainMutation(Mutation):
    """
    Mutate an individual by applying a strain mutation.

    This wraps the ASE GA ``StrainMutation`` and uses a legacy RNG adapter
    internally to provide the required random number generator interface.

    More info in the `ase_ga repo <https://github.com/dtu-energy/ase-ga>`__.

    :param closest_distances: Minimum allowed interatomic distances used
        to validate the mutated structure.
    :type closest_distances: CustomClosestDistances
    :param n_variable_cell_vectors: Number of cell vectors that may be
        changed by the mutation. Defaults to ``3``.
    :type n_variable_cell_vectors: int
    :param cell_bounds: Allowed bounds for the cell parameters. If
        ``None``, bounds are derived from the parent cell by allowing
        ``±1.0`` Angstrom for ``a``, ``b`` and ``c``. Defaults to ``None``.
    :type cell_bounds: CustomCellBounds | None
    :param stddev: Standard deviation of the strain distribution. Defaults
        to ``0.7``.
    :type stddev: float
    :param max_retries: Maximum number of attempts to produce a valid
        mutation. Defaults to ``100``.
    :type max_retries: int
    :param rng: Random number generator. If ``None``, the base class
        creates one.
    :type rng: None | np.random.Generator
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        n_variable_cell_vectors: int = 3,
        cell_bounds: CustomCellBounds | None = None,
        stddev: float = 0.7,
        max_retries: int = 1,
        rng: None | np.random.Generator = None,
    ):
        super().__init__(
            closest_distances=closest_distances, max_retries=max_retries, rng=rng
        )
        self.n_variable_cell_vectors = n_variable_cell_vectors
        self.cell_bounds = cell_bounds
        self.max_steps = 1
        self.stddev = stddev

    def _perform_mutation(self, individual: Individual) -> Individual | None:
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
            rng=self._legacy_rng,  # type: ignore
        )
        ase_strain.update_scaling_volume([individual])

        offspring = individual
        mutant = ase_strain.mutate(offspring)
        return mutant


class EnlargeMutation(Mutation):
    """
    Deterministically enlarge the unit cell by doubling allowed cell vectors.

    For each cell vector, the mutation checks whether doubling that vector
    keeps the cell within ``cell_bounds``. All allowed vectors are doubled
    simultaneously and the structure is repeated to fill the new cell.

    :param closest_distances: Minimum allowed interatomic distances used
        to validate the mutated structure.
    :type closest_distances: CustomClosestDistances
    :param cell_bounds: Allowed bounds for the resulting cell.
    :type cell_bounds: CustomCellBounds
    :param max_retries: Maximum number of attempts to produce a valid
        mutation. Defaults to ``1``.
    :type max_retries: int
    :param rng: Random number generator passed to the base class.
        Defaults to ``None``.
    :type rng: None | np.random.Generator
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        cell_bounds: CustomCellBounds,
        max_retries: int = 1,
        rng: None | np.random.Generator = None,
    ):
        super().__init__(
            closest_distances=closest_distances, max_retries=max_retries, rng=rng
        )

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

    def _perform_mutation(self, individual: Individual) -> Individual | None:
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
    """
    Deterministically reduce the individual's cell to its Niggli form.

    Applies ``build.niggli_reduce`` to the individual in place and returns
    the same object.

    :param closest_distances: Minimum allowed interatomic distances used
        to validate the mutated structure.
    :type closest_distances: CustomClosestDistances
    :param max_retries: Maximum number of attempts to produce a valid
        mutation. Defaults to ``1``.
    :type max_retries: int
    :param rng: Random number generator passed to the base class.
        Defaults to ``None``.
    :type rng: None | np.random.Generator
    """

    def _perform_mutation(self, individual: Individual) -> Individual | None:
        build.niggli_reduce(individual)
        return individual


class MinimizeTiltMutation(Mutation):
    """
    Deterministically minimize the tilt of the individual's cell.

    Applies ``build.minimize_tilt`` to the individual in place and returns
    the same object.

    :param closest_distances: Minimum allowed interatomic distances used
        to validate the mutated structure.
    :type closest_distances: CustomClosestDistances
    :param max_retries: Maximum number of attempts to produce a valid
        mutation. Defaults to ``1``.
    :type max_retries: int
    :param rng: Random number generator passed to the base class.
        Defaults to ``None``.
    :type rng: None | np.random.Generator
    """

    def _perform_mutation(self, individual: Individual) -> Individual | None:
        build.minimize_tilt(individual)
        return individual


class RotationMutation(Mutation):
    """
    Rotate the individual by a random angle around a random Cartesian axis.

    The rotation angle is drawn uniformly from ``[0, 90]`` degrees and the
    axis is chosen uniformly from ``x``, ``y`` or ``z``. The cell itself is
    not rotated.

    :param closest_distances: Minimum allowed interatomic distances used
        to validate the mutated structure.
    :type closest_distances: CustomClosestDistances
    :param max_retries: Maximum number of attempts to produce a valid
        mutation. Defaults to ``1``.
    :type max_retries: int
    :param rng: Random number generator passed to the base class.
        Defaults to ``None``.
    :type rng: None | np.random.Generator
    """

    def _perform_mutation(self, individual: Individual) -> Individual | None:
        v_rand = self._rng.choice(["x", "y", "z"])
        a_rand = self._rng.uniform(0, 90)
        individual.rotate(a=a_rand, v=v_rand, rotate_cell=True)
        return individual
