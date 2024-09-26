import warnings
from abc import ABC, abstractmethod
import numpy as np
import ase
from icecream import ic
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances

# ╔══════════════════════════════════════════════════════════╗
# ║            Abstract Base Class for Mutations             ║
# ╚══════════════════════════════════════════════════════════╝


class Mutation(ABC):
    """
    Here we define all attributes and methods that we need
    for _every_ mutation.
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        max_steps: int = 10,
    ):
        self.closest_distances = closest_distances
        self.max_steps = max_steps

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
    def perform_mutation(self, crystal: ase.Atoms) -> ase.Atoms | None:
        """
        Should calculate the offspring from parent, depending on mutation type.
        If this was not possible, return None.
        """
        pass

    def mutate(self, crystal: ase.Atoms) -> tuple[ase.Atoms, bool]:
        """
        Should calculate the offspring from parent, depending on mutation type.
        Returns the offspring and a boolean if the mutation was successful.
        True if successful, False if not.
        """
        ic("Performing {}.".format(self.__class__.__name__))
        try:
            if not hasattr(self, "max_steps") or self.max_steps == 0:
                self.max_steps = 1

            offspring = None

            keep_offspring = False
            step = 0
            for step in range(self.max_steps):
                offspring = crystal.copy()
                try:
                    offspring = self.perform_mutation(offspring)

                except Exception as e:
                    warnings.warn(
                        "{}: Unknown Error. No mutation possible. {}".format(
                            self.__class__.__name__, e)
                    )
                    keep_offspring = False
                    continue

                if offspring is None:
                    keep_offspring = False
                    continue
                else:
                    offspring.wrap()

                try:
                    offspring_is_valid = self.crystal_is_valid_object(
                        offspring)
                except Exception as e:
                    warnings.warn(
                        "{}: Unknown Error crystal_is_valid_object. {}".format(
                            self.__class__.__name__, e)
                    )
                    keep_offspring = False
                    continue

                try:
                    offspring_is_physical = self.crystal_is_physical(offspring)
                except Exception as e:
                    warnings.warn(
                        "{}: Unknown Error in crystal_is_physical. {}".format(
                            self.__class__.__name__, e)
                    )
                    keep_offspring = False
                    continue

                if not offspring_is_valid:
                    warnings.warn(
                        "{}: Offspring is not a valid object.".format(
                            self.__class__.__name__
                        ) + f"\nOffspring: {offspring}"
                    )
                    keep_offspring = False

                elif not offspring_is_physical:
                    keep_offspring = False

                else:
                    keep_offspring = True
                    break

            if keep_offspring and offspring is not None:
                offspring.wrap()
                offspring_cell = offspring.get_cell()

                # replace all Atoms in parent with offspring
                # Lables and attributes stay the same
                del crystal[:]
                crystal.extend(offspring)
                crystal.set_cell(offspring_cell)
                crystal.set_pbc([True, True, True])

                ic("Done! After {} steps.".format(step+1))
                return crystal, True

            else:
                ic("Mutation failed.")
                return crystal, False

        except Exception as e:
            warnings.warn(
                "{}: Unknown Error. Couldnt perform mutation. {}".format(
                    self.__class__.__name__, e)
            )
            return crystal, False
