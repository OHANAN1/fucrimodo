import logging
from abc import ABC, abstractmethod

import ase
import numpy as np
from ase import build
from ase.build import stack
from ase.cell import Cell
from ase_ga.cutandsplicepairing import CutAndSplicePairing

from ...core import Individual
from ...core.utils import CustomCellBounds, CustomClosestDistances
from ..utils import convert_ase_atoms_to_individual
from ..utils import LegacyRNGAdapter

# ╔══════════════════════════════════════════════════════════╗
# ║                 Abstract Crossover Class                 ║
# ╚══════════════════════════════════════════════════════════╝


class Crossover(ABC):
    """
    Here we define all attributes and methods that we need
    for _every_ crossover. The Crossover class will copy the parents,
    so that the original population is never changed.

    Please always initialize the super class in children, so rng, max retries and closest distances is set.
    For random operation please use the :attr:`_rng`.
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        max_retries: int = 1,
        rng: np.random.Generator | None = None,
    ):
        if not rng:
            rng = np.random.default_rng()
        self._rng = rng
        self.closest_distances = closest_distances
        self.max_retries = max_retries

    @property
    def logger(self) -> logging.Logger | None:
        if not hasattr(self, "_logger"):
            return None
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    def __repr__(self):
        class_name = self.__class__.__name__
        variables = vars(self)

        variables_str = " "
        for key, value in variables.items():
            if key == "closest_distances" or key == "cell_bounds":
                continue

            variables_str += f"{key}={value}, "

        variables_str = variables_str[:-2]
        return f"{class_name}({variables_str})"

    def _individual_is_valid_object(self, individual: Individual) -> bool:
        """
        Tests if the individual is a valid Individual object.
        """
        if not isinstance(individual, Individual):
            return False

        if not len(individual) > 0:
            return False

        if not all(isinstance(atom, ase.Atom) for atom in individual):
            return False

        if np.isnan(individual.get_positions()).any():
            return False

        if np.isnan(individual.get_cell()).any():
            return False

        if np.isnan(individual.get_atomic_numbers()).any():
            return False

        return True

    def _individual_is_physical(self, individual: Individual) -> bool:
        """
        Tests if the individual is physical.
        """
        if individual.cell:
            if individual.get_volume() < 1.0:
                return False

        if self.closest_distances.atoms_are_too_close(individual):
            return False

        return True

    @abstractmethod
    def _perform_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:
        """
        This is the methode where the specific crossover is performed.
        No checks are done here, only the crossover.

        If None is returned for one of the individuals the :meth:`crossover`
        will return a copy of the original individual.
        """
        pass

    def crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual, bool]:
        """
        Should calculate the offsprings from parents,
        depending on crossover type.
        Returns the two offsprings and a boolean if the crossover was successful.
        True if successful, False if not.
        """
        if self.logger:
            self.logger.debug("Performing {}.".format(self.__class__.__name__))

        offspring_1 = None
        offspring_2 = None

        par1_pbc = parent1.pbc.copy()
        par2_pbc = parent2.pbc.copy()

        keep_offspring = False
        step = 0
        for step in range(self.max_retries):

            offspring_1 = parent1.copy()
            offspring_2 = parent2.copy()

            offspring_1, offspring_2 = self._perform_crossover(offspring_1, offspring_2)

            if offspring_1 is None or offspring_2 is None:
                keep_offspring = False
                continue

            offspring_1.wrap()
            offspring_2.wrap()

            # Check if all params for ase atoms obj are fullfilled
            of_1_is_valid = self._individual_is_valid_object(offspring_1)
            of_2_is_valid = self._individual_is_valid_object(offspring_2)
            offspring_is_valid = of_1_is_valid and of_2_is_valid
            if not offspring_is_valid:
                if self.logger:
                    self.logger.warning(
                        f"{self.__class__.__name__}: Offspring is not a valid object."
                        + f"\nOffspring: {offspring_1} or {offspring_2}"
                    )
                keep_offspring = False
                continue

            # Check if minimum physical requirements are met
            of_1_is_physical = self._individual_is_physical(offspring_1)
            of_2_is_physical = self._individual_is_physical(offspring_2)
            offspring_is_physical = of_1_is_physical and of_2_is_physical
            if not offspring_is_physical:
                if self.logger:
                    self.logger.warning(
                        f"{self.__class__.__name__}: Offspring is not a physically feasable."
                        + f"\nOffspring: {offspring_1} or {offspring_2}"
                    )
                keep_offspring = False
                continue

            keep_offspring = True

        if keep_offspring and offspring_1 and offspring_2:
            offspring_1.wrap()
            offspring_2.wrap()

            offspring_1_cell = offspring_1.get_cell()
            offspring_2_cell = offspring_2.get_cell()

            # replace all Atoms in parent with offspring
            # Lables and attributes stay the same
            del parent1[:]
            parent1.extend(offspring_1)
            parent1.set_cell(offspring_1_cell)
            parent1.set_pbc(par1_pbc)

            del parent2[:]
            parent2.extend(offspring_2)
            parent2.set_cell(offspring_2_cell)
            parent2.set_pbc(par2_pbc)

            if self.logger:
                self.logger.debug("Done! After {} steps.".format(step + 1))
            return (parent1, parent2, True)

        else:
            if self.logger:
                self.logger.debug("Crossover failed.")
            return (parent1, parent2, False)


# ╔══════════════════════════════════════════════════════════╗
# ║                    Crossover Classes                     ║
# ╚══════════════════════════════════════════════════════════╝


class UnitCellCrossover(Crossover):
    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        scale_atoms: bool = True,
        max_retries: int = 10,
        rng: None | np.random.Generator = None,
    ):
        super().__init__(
            closest_distances=closest_distances, max_retries=max_retries, rng=rng
        )
        self.scale_atoms = scale_atoms

    def _perform_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:

        offspring1 = parent1
        offspring2 = parent2

        cell1 = parent1.get_cell()[:]  # type: ignore
        cell2 = parent2.get_cell()[:]  # type: ignore

        if np.all(cell1 == cell2):
            return (None, None)

        cell_v1 = [cell1[0], cell2[0]]
        self._rng.shuffle(cell_v1)

        cell_v2 = [cell1[1], cell2[1]]
        self._rng.shuffle(cell_v2)

        cell_v3 = [cell1[2], cell2[2]]
        self._rng.shuffle(cell_v3)

        new_cell1 = Cell([cell_v1[0], cell_v2[0], cell_v3[0]])

        new_cell2 = Cell([cell_v1[1], cell_v2[1], cell_v3[1]])

        offspring1.set_cell(new_cell1, scale_atoms=self.scale_atoms)
        offspring2.set_cell(new_cell2, scale_atoms=self.scale_atoms)

        return (offspring1, offspring2)


class StackCellsCrossover(Crossover):
    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        cell_bounds: CustomCellBounds,
        scale_atoms: bool = True,
        max_retries: int = 10,
        rng: None | np.random.Generator = None,
    ):
        super().__init__(
            closest_distances=closest_distances, max_retries=max_retries, rng=rng
        )
        self.scale_atoms = scale_atoms
        self.cell_bounds = cell_bounds

    def _perform_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:

        axis = self._rng.integers(0, 2)

        cell1 = parent1.get_cell()[:]  # type: ignore
        cell2 = parent2.get_cell()[:]  # type: ignore
        cell1[axis] += cell2[axis]
        new_cell = Cell(cell1)

        if not self.cell_bounds.is_within_bounds(new_cell):
            return (None, None)

        offspring1 = stack(parent1, parent2, axis=axis, cell=new_cell, maxstrain=None)  # type: ignore
        offspring2 = stack(parent2, parent1, axis=axis, cell=new_cell, maxstrain=None)  # type: ignore

        if isinstance(offspring1, Individual) and isinstance(offspring2, Individual):
            return (offspring1, offspring2)
        else:
            return (None, None)


class OnePointElementCrossover(Crossover):
    def _perform_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:

        # Get minimum length of both parents
        min_length = min(len(parent1), len(parent2))

        # If one of the parents has only one element, crossover is not possible
        if min_length <= 1:
            return (None, None)

        # Get atomic numbers of both parents
        par_1_atomic_numbers = parent1.get_atomic_numbers().tolist()
        par_2_atomic_numbers = parent2.get_atomic_numbers().tolist()

        # Check if only one element is present in both parents
        if len(set(par_1_atomic_numbers + par_2_atomic_numbers)) == 1:
            return (None, None)

        # Get random index where to split the atomic numbers
        # Must be smaller then the minimum length to avoid to
        # many atoms in the offspring
        if min_length <= 1:
            cut_index = min_length
        else:
            cut_index = self._rng.integers(1, min_length)

        # Get the cut of atomic numbers of the opposite parents and
        # concatenate them to get the new atomic numbers
        new_par_1_numbers = (
            par_2_atomic_numbers[:cut_index] + par_1_atomic_numbers[cut_index:]
        )
        new_par_2_numbers = (
            par_1_atomic_numbers[:cut_index] + par_2_atomic_numbers[cut_index:]
        )

        offspring1 = parent1.copy()
        offspring1.set_atomic_numbers(new_par_1_numbers)
        offspring1.wrap()

        offspring2 = parent2.copy()
        offspring2.set_atomic_numbers(new_par_2_numbers)
        offspring2.wrap()

        return (offspring1, offspring2)


class OnePointPositionCrossover(Crossover):
    def _perform_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:

        # Get minimum length of both parents
        min_length = min(len(parent1), len(parent2))

        # If one of the parents has only one element, crossover is not possible
        if min_length < 2:
            return (None, None)

        # Get atomic numbers of both parents
        par_1_atomic_pos = parent1.get_positions().tolist()
        par_2_atomic_pos = parent2.get_positions().tolist()

        # Get random index where to split the atomic numbers
        # Must be smaller then the minimum length to avoid to
        # many atoms in the offspring
        if min_length <= 1:
            cut_index = min_length
        else:
            cut_index = self._rng.integers(1, min_length)

        # Get the cut of atomic numbers of the opposite parents and
        # concatenate them to get the new atomic numbers
        new_par_1_pos = par_2_atomic_pos[:cut_index] + par_1_atomic_pos[cut_index:]
        new_par_2_pos = par_1_atomic_pos[:cut_index] + par_2_atomic_pos[cut_index:]

        offspring1 = parent1.copy()
        offspring1.set_positions(np.array(new_par_1_pos))
        offspring1.wrap()

        offspring2 = parent2.copy()
        offspring2.set_positions(np.array(new_par_2_pos))
        offspring2.wrap()

        return (offspring1, offspring2)


class CutAndSpliceCrossover(Crossover):
    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        cell_bounds: CustomCellBounds,
        n_top: int | str = "all",
        number_of_variable_cell_vectors: int = 0,
        max_retries: int = 10,
        rng: None | np.random.Generator = None,
    ):
        super().__init__(
            closest_distances=closest_distances, max_retries=max_retries, rng=rng
        )

        self.cell_bounds = cell_bounds
        self.n_top = n_top
        self.cell_bounds = cell_bounds
        self.number_of_variable_cell_vectors = number_of_variable_cell_vectors

    def _perform_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:
        if self.n_top == "all":
            n_top = len(parent1)
        else:
            n_top = self.n_top

        # sort parents, since the stoichometry is not the same if not sorted
        # and the CutAndSplicePairing will not work
        parent1 = build.sort(parent1)
        parent2 = build.sort(parent2)

        # Check if the parents have the same stoichiometry since this is
        # necessary for the CutAndSplicePairing of ase
        # Return None if it is not the case, to signal that the crossover failed
        if not np.array_equal(parent1.numbers, parent2.numbers):
            return (None, None)

        # Check if the parents are of the same length, this is necessary
        # for the CutAndSplicePairing of ase
        if len(parent1) != len(parent2):
            return (None, None)

        # Check if the parents have a minimum of 2 atoms, else the
        # CutAndSplicePairing is unnecessary
        if len(parent1) < 2:
            return (None, None)

        # If the cell vectors are not the same and not variable
        # They can not be crossed. Return None in this case
        cell1 = parent1.get_cell()
        cell2 = parent2.get_cell()
        for i in range(self.number_of_variable_cell_vectors, 3):
            if not np.allclose(cell1[i], cell2[i]):  # type: ignore
                return (None, None)

        cut_and_splice_pairing = CutAndSplicePairing(
            slab=ase.Atoms(),
            blmin=self.closest_distances,
            n_top=n_top,
            cellbounds=self.cell_bounds,
            number_of_variable_cell_vectors=self.number_of_variable_cell_vectors,
            # Use the old api throught the adapter
            rng=LegacyRNGAdapter(self._rng),  # type: ignore
        )

        # Create the first offspring
        offspring_1 = cut_and_splice_pairing.cross(parent1, parent2)

        # If it is not possible to create offspring, return None
        if offspring_1 is None:
            return (None, None)

        # Create the second offspring
        offspring_2 = cut_and_splice_pairing.cross(parent2, parent1)

        # If it is not possible to create offspring, return None
        if offspring_2 is None:
            return (None, None)

        # If both offspring are valid, return them
        return (
            convert_ase_atoms_to_individual(offspring_1),
            convert_ase_atoms_to_individual(offspring_2),
        )
