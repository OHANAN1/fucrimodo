from abc import ABC, abstractmethod
import numpy as np
import ase
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.modules import Individual
import logging
from copy import deepcopy

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

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, "_logger"):
            raise AttributeError(
                f"{self.__class__.__name__}: No logger set. Please set a logger."
            )
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

    def individual_is_valid_object(self, individual: Individual) -> bool:
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

    def individual_is_physical(self, individual: Individual) -> bool:
        """
        Tests if the individual is physical.
        """
        if individual.get_volume() < 1.0:
            return False

        if self.closest_distances.atoms_are_too_close(individual):
            return False

        return True

    @abstractmethod
    def perform_mutation(self, individual: Individual) -> Individual | None:
        """
        Should calculate the offspring from parent, depending on mutation type.
        If this was not possible, return None.
        """
        pass

    def mutate(self, individual: Individual) -> tuple[Individual, bool]:
        """
        Should calculate the offspring from parent, depending on mutation type.
        Returns the offspring and a boolean if the mutation was successful.
        True if successful, False if not.
        """
        self.logger.debug("Performing {}.".format(self.__class__.__name__))
        try:
            if not hasattr(self, "max_steps") or self.max_steps == 0:
                self.max_steps = 1

            offspring = None
            if hasattr(individual, "constraints"):
                constraints = deepcopy(individual.constraints)
            else:
                constraints = []

            keep_offspring = False
            step = 0
            for step in range(self.max_steps):
                offspring = individual.copy()
                offspring.constraints = constraints
                try:
                    offspring = self.perform_mutation(offspring)

                except Exception as e:
                    self.logger.warning(
                        "{}: Unknown Error. No mutation possible. {}".format(
                            self.__class__.__name__, e
                        )
                    )
                    keep_offspring = False
                    continue

                if offspring is None:
                    keep_offspring = False
                    continue
                else:
                    offspring.wrap()

                try:
                    offspring_is_valid = self.individual_is_valid_object(offspring)
                except Exception as e:
                    self.logger.warning(
                        "{}: Unknown Error individual_is_valid_object. {}".format(
                            self.__class__.__name__, e
                        )
                    )
                    keep_offspring = False
                    continue

                try:
                    offspring_is_physical = self.individual_is_physical(offspring)
                except Exception as e:
                    self.logger.warning(
                        "{}: Unknown Error in individual_is_physical. {}".format(
                            self.__class__.__name__, e
                        )
                    )
                    keep_offspring = False
                    continue

                if not offspring_is_valid:
                    self.logger.warning(
                        "{}: Offspring is not a valid object.".format(
                            self.__class__.__name__
                        )
                        + f"\nOffspring: {offspring}"
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
                del individual[:]
                individual.extend(offspring)

                # make sure constraints are set
                individual.set_constraint(constraints)
                individual.set_cell(offspring_cell)
                individual.set_pbc([True, True, True])

                self.logger.debug("Done! After {} steps.".format(step + 1))
                return individual, True

            else:
                self.logger.debug("Mutation failed.")
                return individual, False

        except Exception as e:
            self.logger.error(
                "{}: Unknown Error. Couldnt perform mutation. {}".format(
                    self.__class__.__name__, e
                )
            )
            return individual, False
