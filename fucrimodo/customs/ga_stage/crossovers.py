from typing import Callable
import random
from ase.cell import Cell
import numpy as np
from abc import ABC, abstractmethod
import ase
from ase.ga.utilities import closest_distances_generator
import warnings
from numpy.typing import NDArray

from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
import ase.ga.element_crossovers as ase_elem_cross
import ase.ga.cutandsplicepairing as ase_cut_and_splice
from ase.build import stack, cut
from ase.build import attach
from ase.visualize import view
import concurrent.futures
from fucrimodo.core.modules import Individual

from ase.geometry import wrap_positions
from ase.geometry import get_distances
from ase.build import make_supercell

from icecream import ic

import numpy as np
from abc import ABC, abstractmethod
import ase
import warnings
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from typing import Callable
import concurrent.futures
from icecream import ic



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
            print("Funktion hat zu lange gedauert und wurde abgebrochen")
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
        print("Performing {}.".format(self.__class__.__name__))

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
                warnings.warn(
                    "{}: Unknown Error. No mutation possible. {}".format(
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
                    warnings.warn(
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
                warnings.warn(
                    "{}: Unknown Error in crystal_is_valid_object. {}".format(
                        self.__class__.__name__, e)
                )
                keep_offspring = False
                continue

            try:
                offspring_1_is_physical = self.crystal_is_physical(offspring_1)
                offspring_2_is_physical = self.crystal_is_physical(offspring_2)
            except Exception as e:
                warnings.warn(
                    "{}: Unknown Error in crystal_is_physical. {}".format(
                        self.__class__.__name__, e)
                )
                keep_offspring = False
                continue

            if not offspring_1_is_valid or not offspring_2_is_valid:
                warnings.warn(
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

                print("Done! After {} steps.".format(step+1))
                return (parent1, parent2, True)

            else:
                print("Crossover failed.")
                return (parent1, parent2, False)

        except Exception as e:
            warnings.warn(
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

        min_length = min(len(parent1), len(parent2))

        par_1_atomic_numbers = parent1.get_atomic_numbers()
        par_2_atomic_numbers = parent2.get_atomic_numbers()

        all_atomic_numbers = np.unique(
            np.concatenate([par_1_atomic_numbers, par_2_atomic_numbers])
        )
        if len(all_atomic_numbers) == 1:
            return (None, None)

        par_1_start_index = 0
        if len(par_1_atomic_numbers) > min_length:
            par_1_start_index = random.randint(0, len(par_1_atomic_numbers) - min_length)

        selected_par_1_numbers = par_1_atomic_numbers[
            par_1_start_index:par_1_start_index + min_length
        ]

        par_2_start_index = 0
        if len(par_2_atomic_numbers) > min_length:
            par_2_start_index = random.randint(0, len(par_2_atomic_numbers) - min_length)

        selected_par_2_numbers = par_2_atomic_numbers[
            par_2_start_index:par_2_start_index + min_length
        ]

        if min_length > 2:
            cut_index = random.randint(1, min_length - 1)

            new_par_1_numbers = np.concatenate(
                [
                    selected_par_1_numbers[:cut_index],
                    selected_par_2_numbers[cut_index:]
                ]
            )

            new_par_2_numbers = np.concatenate(
                [
                    selected_par_2_numbers[:cut_index],
                    selected_par_1_numbers[cut_index:]
                ]
            )
        else:
            new_par_1_numbers = selected_par_2_numbers
            new_par_2_numbers = selected_par_1_numbers

        if par_1_start_index > 0:
            new_par_1_numbers = np.concatenate(
                [
                    par_1_atomic_numbers[:par_1_start_index],
                    new_par_1_numbers
                ]
            )
        if par_2_start_index > 0:
            new_par_2_numbers = np.concatenate(
                [
                    par_2_atomic_numbers[:par_2_start_index],
                    new_par_2_numbers
                ]
            )

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

        min_length = min(len(parent1), len(parent2))

        par_1_positions = parent1.get_positions()
        par_2_positions = parent2.get_positions()


        par_1_start_index = 0
        if len(par_1_positions) > min_length:
            par_1_start_index = random.randint(0, len(par_1_positions) - min_length)

        selected_par_1_positions = par_1_positions[
            par_1_start_index:par_1_start_index + min_length
        ]

        par_2_start_index = 0
        if len(par_2_positions) > min_length:
            par_2_start_index = random.randint(0, len(par_2_positions) - min_length)

        selected_par_2_positions = par_2_positions[
            par_2_start_index:par_2_start_index + min_length
        ]

        if min_length > 2:
            cut_index = random.randint(1, min_length - 1)

            new_par_1_positions = np.concatenate(
                [
                    selected_par_1_positions[:cut_index],
                    selected_par_2_positions[cut_index:]
                ]
            )

            new_par_2_positions = np.concatenate(
                [
                    selected_par_2_positions[:cut_index],
                    selected_par_1_positions[cut_index:]
                ]
            )
        else:
            new_par_1_positions = selected_par_2_positions
            new_par_2_positions = selected_par_1_positions

        if par_1_start_index > 0:
            new_par_1_positions = np.concatenate(
                [
                    par_1_positions[:par_1_start_index],
                    new_par_1_positions
                ]
            )
        if par_2_start_index > 0:
            new_par_2_positions = np.concatenate(
                [
                    par_2_positions[:par_2_start_index],
                    new_par_2_positions
                ]
            )

        offspring1 = parent1.copy()
        offspring1.set_positions(new_par_1_positions)
        offspring1.wrap()

        offspring2 = parent2.copy()
        offspring2.set_positions(new_par_2_positions)
        offspring2.wrap()

        return (offspring1, offspring2)




class CutAndSpliceCrossover(Crossover):

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        cell_bounds: CustomCellBounds,
        max_steps: int = 2,
        max_atoms_to_cut: int = 5,
    ):
        self.max_steps = max_steps
        self.closest_distances = closest_distances
        self.cell_bounds = cell_bounds
        self.max_atoms_to_cut = max_atoms_to_cut

    def __get_slap_from_parent(
        self,
        parent: Individual,
        a_len: float = 10.,
        b_len: float = 10.
    ) -> Individual | None:
        slab = cut(
            parent, a=(a_len, 0, 0), b=(0, b_len, 0), tolerance=0.1,
            maxatoms=self.max_atoms_to_cut, nlayers=1
        )
        if len(slab) == 0:
            return None
        else:
            return slab

    def __timed_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:

        min_length = min(len(parent1), len(parent2))
        if min_length < self.max_atoms_to_cut:
            return (None, None)

        par1 = parent1.copy()
        par2 = parent2.copy()

        cell_par1 = par1.cell.cellpar()
        cell_par2 = par2.cell.cellpar()

        a_len = min([cell_par1[0], cell_par2[0]])
        a_lens = [a_len/cell_par1[0], a_len/cell_par2[0]]

        b_len = min([cell_par1[1], cell_par2[1]])
        b_lens = [b_len/cell_par1[1], b_len/cell_par2[1]]

        slab1 = self.__get_slap_from_parent(
            par1,  a_len=a_lens[0], b_len=b_lens[0]
        )
        slab2 = self.__get_slap_from_parent(
            par2, a_len=a_lens[1], b_len=b_lens[1]
        )

        offspring1 = stack(
            slab1, slab2, axis=2
        )
        offspring2 = stack(
            slab2, slab1, axis=2
        )

        if offspring1 is None or offspring2 is None:
            return (None, None)
        else:
            if (
                self.cell_bounds.is_within_bounds(
                    offspring1.get_cell()  # type: ignore
                )
                and self.cell_bounds.is_within_bounds(
                    offspring2.get_cell())  # type: ignore
            ):
                return (offspring1, offspring2)  # type: ignore

            else:
                return (None, None)

    def perform_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:

        result = run_function_with_timeout(
            lambda: self.__timed_crossover(parent1, parent2),
            timeout=30
        )
        if result is None:
            print("Crossover took too long. I dont know why tho")
            print("Crossover took too long. I dont know why tho")

            return (None, None)

        return result


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
