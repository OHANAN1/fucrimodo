from typing import Callable
import random
from ase.cell import Cell
import numpy as np
from abc import ABC, abstractmethod
import ase
from numpy.typing import NDArray
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from ase.build import stack, cut
import concurrent.futures
from fucrimodo.core.modules import Individual
from ase.geometry import get_distances
import ase
from ase import build
from typing import Callable
import concurrent.futures
from ase.ga.cutandsplicepairing import CutAndSplicePairing

import logging

from fucrimodo.customs.population_generator import convert_ase_atoms_to_individual
logger = logging.getLogger('run_logger')

# ╔══════════════════════════════════════════════════════════╗
# ║                    Utility functions                     ║
# ╚══════════════════════════════════════════════════════════╝

def atoms_in_crystal_are_to_close(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    cell: Cell,
    closest_distances: dict[tuple[int, int], float],
    n_neighbours_to_check: int = 10
) -> bool:
    """
    Returns True if atoms in crystal are to close to each other.
    """
    n_atoms = len(atomic_numbers)

    for i in range(n_atoms):
        _, distances = get_distances(
            positions,
            positions[i],
            cell=cell,
            pbc=True
        )

        closest_neighbours = np.argsort(
            distances,
            axis=0
        )[1:n_neighbours_to_check].flatten()

        for j in closest_neighbours:
            if i == j:
                continue

            atomic_n_i: int = atomic_numbers[i]
            atomic_n_j: int = atomic_numbers[j]  # type: ignore

            min_allowed_distance = closest_distances[(atomic_n_i, atomic_n_j)]
            current_distance = distances[j]

            if current_distance < min_allowed_distance:
                return True

    return False


def adjust_atoms_positions(
    positions: NDArray[np.float64],
    atomic_numbers: NDArray[np.int64],
    cell: Cell,
    closest_distances: dict,
    n_neighbours_to_check: int = 10,
) -> None:
    """
    Adjusts atoms positions in the crystal to ensure minimum distances are
    maintained.
    """
    n_atoms = len(atomic_numbers)
    atomic_numbers = atomic_numbers.tolist()

    for i in range(n_atoms):
        _, distances = get_distances(
            positions,
            positions[i],
            cell=cell,
            pbc=True
        )

        closest_neighbours = np.argsort(
            distances,
            axis=0
        )[1:n_neighbours_to_check].flatten()

        for j in closest_neighbours.tolist():
            if i == j:
                continue

            atomic_n_i = atomic_numbers[i]
            atomic_n_j = atomic_numbers[j]

            min_allowed_distance = closest_distances[
                (atomic_n_i, atomic_n_j)
            ]
            current_distance = distances[j]

            if current_distance < min_allowed_distance:
                direction = positions[j] - positions[i]
                direction /= np.linalg.norm(direction)

                correction = direction * \
                    (min_allowed_distance - current_distance)
                positions[i] -= correction
                positions[j] += correction


# ╔══════════════════════════════════════════════════════════╗
# ║                    Utility functions                     ║
# ╚══════════════════════════════════════════════════════════╝

def run_function_with_timeout(funktion: Callable, timeout: int = 60):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(funktion)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.error("Funktion hat zu lange gedauert und wurde abgebrochen")
            return None

# ╔══════════════════════════════════════════════════════════╗
# ║                 Abstract Crossover Class                 ║
# ╚══════════════════════════════════════════════════════════╝

class Crossover(ABC):
    """
    Here we define all attributes and methods that we need
    for _every_ crossover. The Crossover class will copy the parents,
    so that the original population is never changed.
    """

    def __init__(self, closest_distances: CustomClosestDistances):
        self.closest_distances = closest_distances

    def __repr__(self):
        class_name = self.__class__.__name__
        variables = vars(self)

        variables_str = ' '
        for key, value in variables.items():
            if key == "closest_distances" or key == "cell_bounds":
                continue

            variables_str += f'{key}={value}, '

        variables_str = variables_str[:-2]
        return f'{class_name}({variables_str})'

    def crystal_is_valid_object(self, crystal: Individual) -> bool:
        """
        Tests if the crystal is a valid Individual object.
        """
        if not isinstance(crystal, Individual):
            return False

        if not len(crystal) > 0:
            return False

        if not all(isinstance(atom, ase.Atom) for atom in crystal):
            return False

        if np.isnan(crystal.get_positions()).any():
            return False

        if np.isnan(crystal.get_cell()).any():
            return False

        if np.isnan(crystal.get_atomic_numbers()).any():
            return False

        return True

    def crystal_is_physical(self, crystal: Individual) -> bool:
        """
        Tests if the crystal is physical.
        """
        if crystal.get_volume() < 1.:
            return False

        if self.closest_distances.atoms_are_too_close(crystal):
            return False

        return True

    @abstractmethod
    def perform_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:
        """
        This is the methode where the specific crossover is performed.
        No checks are done here, only the crossover.
        """
        pass

    def crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> tuple[Individual, Individual, bool]:
        """
        Should calculate the offsprings from parents,
        depending on crossover type.
        Returns the two offsprings and a boolean if the crossover was successful.
        True if successful, False if not.
        """
        logger.info("Performing {}.".format(self.__class__.__name__))

        if not hasattr(self, "max_steps") or self.max_steps == 0:
            self.max_steps = 1

        offspring_1 = None
        offspring_2 = None

        keep_offspring = False
        step = 0
        for step in range(self.max_steps):
            offspring_1 = parent1.copy()
            offspring_2 = parent2.copy()

            try:
                offspring_1, offspring_2 = self.perform_crossover(
                    offspring_1, offspring_2
                )

            except Exception as e:
                logger.error(
                    "{}: Unknown Error. No crossover possible. {}".format(
                        self.__class__.__name__, e)
                )
                keep_offspring = False
                continue

            if offspring_1 is None or offspring_2 is None:
                keep_offspring = False
                continue
            else:
                try:
                    offspring_1.wrap()
                    offspring_2.wrap()
                except Exception as e:
                    logger.error(
                        "{}: Unknown Error in wrapping. {}".format(
                            self.__class__.__name__, e)
                    )
                    keep_offspring = False
                    continue

            try:
                offspring_1_is_valid = self.crystal_is_valid_object(
                    offspring_1
                )
                offspring_2_is_valid = self.crystal_is_valid_object(
                    offspring_2
                )
            except Exception as e:
                logger.error(
                    "{}: Unknown Error in crystal_is_valid_object. {}".format(
                        self.__class__.__name__, e)
                )
                keep_offspring = False
                continue

            try:
                offspring_1_is_physical = self.crystal_is_physical(offspring_1)
                offspring_2_is_physical = self.crystal_is_physical(offspring_2)
            except Exception as e:
                logger.error(
                    "{}: Unknown Error in crystal_is_physical. {}".format(
                        self.__class__.__name__, e)
                )
                keep_offspring = False
                continue

            if not offspring_1_is_valid or not offspring_2_is_valid:
                logger.warning(
                    "{}: Offspring is not a valid object.".format(
                        self.__class__.__name__
                    ) + f"\nOffspring: {offspring_1} or {offspring_2}"
                )
                keep_offspring = False

            elif not offspring_1_is_physical or not offspring_2_is_physical:
                keep_offspring = False

            else:
                keep_offspring = True
                break

        try:

            if (
                keep_offspring and
                offspring_1 is not None
                and offspring_2 is not None
            ):
                offspring_1.wrap()
                offspring_2.wrap()

                offspring_1_cell = offspring_1.get_cell()
                offspring_2_cell = offspring_2.get_cell()

                # replace all Atoms in parent with offspring
                # Lables and attributes stay the same
                del parent1[:]
                parent1.extend(offspring_1)
                parent1.set_cell(offspring_1_cell)
                parent1.set_pbc([True, True, True])

                del parent2[:]
                parent2.extend(offspring_2)
                parent2.set_cell(offspring_2_cell)
                parent2.set_pbc([True, True, True])

                logger.info("Done! After {} steps.".format(step+1))
                return (parent1, parent2, True)

            else:
                logger.info("Crossover failed.")
                return (parent1, parent2, False)

        except Exception as e:
            logger.error(
                "{}: Unknown Error. Couldnt return offspring. {}".format(
                    self.__class__.__name__, e)
            )
            return (parent1, parent2, False)

# ╔══════════════════════════════════════════════════════════╗
# ║                    Crossover Classes                     ║
# ╚══════════════════════════════════════════════════════════╝


class UnitCellCrossover(Crossover):
    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        scale_atoms: bool = True,
        max_steps: int = 10
    ):
        self.scale_atoms = scale_atoms
        self.max_steps = max_steps
        self.closest_distances = closest_distances

    def perform_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:

        offspring1 = parent1
        offspring2 = parent2

        cell1 = parent1.get_cell()[:]  # type: ignore
        cell2 = parent2.get_cell()[:]  # type: ignore

        if np.all(cell1 == cell2):
            return (None, None)

        cell_v1 = [cell1[0], cell2[0]]
        np.random.shuffle(cell_v1)

        cell_v2 = [cell1[1], cell2[1]]
        np.random.shuffle(cell_v2)

        cell_v3 = [cell1[2], cell2[2]]
        np.random.shuffle(cell_v3)

        new_cell1 = Cell(
            [cell_v1[0], cell_v2[0], cell_v3[0]]
        )

        new_cell2 = Cell(
            [cell_v1[1], cell_v2[1], cell_v3[1]]
        )

        offspring1.set_cell(new_cell1, scale_atoms=self.scale_atoms)
        offspring2.set_cell(new_cell2, scale_atoms=self.scale_atoms)

        return (offspring1, offspring2)


class StackCellsCrossover(Crossover):
    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        cell_bounds: CustomCellBounds,
        scale_atoms: bool = True,
        max_steps: int = 10
    ):
        self.scale_atoms = scale_atoms
        self.max_steps = max_steps
        self.closest_distances = closest_distances
        self.cell_bounds = cell_bounds

    def perform_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:

        axis = random.randint(0, 2)

        cell1 = parent1.get_cell()[:]  # type: ignore
        cell2 = parent2.get_cell()[:]  # type: ignore
        cell1[axis] += cell2[axis]
        new_cell = Cell(cell1)

        if not self.cell_bounds.is_within_bounds(new_cell):
            return (None, None)

        offspring1 = stack(
            parent1,
            parent2,
            axis=axis,
            cell=new_cell
        )
        offspring2 = stack(
            parent2,
            parent1,
            axis=axis,
            cell=new_cell
        )

        if (
            isinstance(offspring1, Individual)
            and isinstance(offspring2, Individual)
        ):
            return (offspring1, offspring2)
        else:
            return (None, None)


class OnePointElementCrossover(Crossover):

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        max_steps: int = 10,
    ):
        self.max_steps = max_steps
        self.closest_distances = closest_distances

    def perform_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:

        # Get minimum length of both parents
        min_length = min(len(parent1), len(parent2))

        # If one of the parents has only one element, crossover is not possible
        if min_length < 2:
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
        cut_index = random.randint(0, min_length - 1)

        # Get the cut of atomic numbers of the opposite parents and
        # concatenate them to get the new atomic numbers
        new_par_1_numbers = par_2_atomic_numbers[:cut_index] + par_1_atomic_numbers[cut_index:]
        new_par_2_numbers = par_1_atomic_numbers[:cut_index] + par_2_atomic_numbers[cut_index:]

        offspring1 = parent1.copy()
        offspring1.set_atomic_numbers(new_par_1_numbers)
        offspring1.wrap()

        offspring2 = parent2.copy()
        offspring2.set_atomic_numbers(new_par_2_numbers)
        offspring2.wrap()

        return (offspring1, offspring2)


class OnePointPositionCrossover(Crossover):

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        max_steps: int = 10,
    ):
        self.max_steps = max_steps
        self.closest_distances = closest_distances

    def perform_crossover(
        self,
        parent1: Individual,
        parent2: Individual
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
        cut_index = random.randint(0, min_length - 1)

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
        max_steps: int = 1,
    ):
        self.max_steps = max_steps
        self.closest_distances = closest_distances
        self.cell_bounds = cell_bounds
        self.n_top = n_top
        self.cell_bounds = cell_bounds

    def perform_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:
        if self.n_top == "all":
            n_top = len(parent1)
        else:
            n_top = self.n_top

        # sort parents, since the stoichometry is not the same if not sorted
        # and the CutAndSplicePairing will not work
        parent1 = build.sort(parent1)
        parent2 = build.sort(parent2)

        cut_and_splice_pairing = CutAndSplicePairing(
            slab=ase.Atoms(),
            blmin=self.closest_distances,
            n_top=n_top,
            cellbounds=self.cell_bounds
        )

        # Create the first offspring
        offspring_1 = cut_and_splice_pairing.cross(
            parent1, parent2
        )

        # If it is not possible to create offspring, return None
        if offspring_1 is None:
            return (None, None)

        # Create the second offspring
        offspring_2 = cut_and_splice_pairing.cross(
            parent2, parent1
        )

        # If it is not possible to create offspring, return None
        if offspring_2 is None:
            return (None, None)

        # If both offspring are valid, return them
        return (
            convert_ase_atoms_to_individual(offspring_1),
            convert_ase_atoms_to_individual(offspring_2)
        )


class DoNothingCrossover(Crossover):
    """
    This Crossover just returns copies of the parents.
    Use crossover_probability in genetic algorithm rather than this
    class to turn off crossover.
    """

    def __init__(self):
        pass

    def perform_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:
        return (None, None)
