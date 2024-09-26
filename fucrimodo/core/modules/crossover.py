import numpy as np
from abc import ABC, abstractmethod
import ase
import warnings
from ..utils.closest_distances_class import CustomClosestDistances
from typing import Callable
import concurrent.futures
from icecream import ic

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
        variables.pop("closest_distances")

        variables_str = ' '
        for key, value in variables.items():
            variables_str += f'{key}={value}, '

        variables_str = variables_str[:-2]
        return f'{class_name}({variables_str})'

    def crystal_is_valid_object(self, crystal: ase.Atoms) -> bool:
        """
        Tests if the crystal is a valid ase.Atoms object.
        """
        if not isinstance(crystal, ase.Atoms):
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

    def crystal_is_physical(self, crystal: ase.Atoms) -> bool:
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
        parent1: ase.Atoms,
        parent2: ase.Atoms
    ) -> tuple[ase.Atoms, ase.Atoms] | tuple[None, None]:
        """
        This is the methode where the specific crossover is performed.
        No checks are done here, only the crossover.
        """
        pass

    def crossover(
        self,
        parent1: ase.Atoms,
        parent2: ase.Atoms
    ) -> tuple[ase.Atoms, ase.Atoms, bool]:
        """
        Should calculate the offsprings from parents,
        depending on crossover type.
        Returns the two offsprings and a boolean if the crossover was successful.
        True if successful, False if not.
        """
        ic("Performing {}.".format(self.__class__.__name__))

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

                ic("Done! After {} steps.".format(step+1))
                return (parent1, parent2, True)

            else:
                ic("Crossover failed.")
                return (parent1, parent2, False)

        except Exception as e:
            warnings.warn(
                "{}: Unknown Error. Couldnt return offspring. {}".format(
                    self.__class__.__name__, e)
            )
            return (parent1, parent2, False)
