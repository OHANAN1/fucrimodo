from abc import ABC, abstractmethod
import numpy as np
import ase
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core import Individual
from ...utils import LegacyRNGAdapter
import logging
from copy import deepcopy

# ╔══════════════════════════════════════════════════════════╗
# ║            Abstract Base Class for Mutations             ║
# ╚══════════════════════════════════════════════════════════╝


class Mutation(ABC):
    """
    Here we define all attributes and methods that we need
    for _every_ mutation.

    Please initiate the super class always with closest_dist, max_retries and rng.

    Note that legacy_rng also needs to be considered for some mutations
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        max_retries: int = 1,
        rng: None | np.random.Generator = None,
    ):
        if not rng:
            rng = np.random.default_rng()
        self._rng = rng
        self._legacy_rng = LegacyRNGAdapter(self._rng)

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

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

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
        if individual.cell:
            if individual.get_volume() < 1.0:
                return False

        if self.closest_distances.atoms_are_too_close(individual):
            return False

        return True

    @abstractmethod
    def _perform_mutation(self, individual: Individual) -> Individual | None:
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
        if self.logger:
            self.logger.debug("Performing {}.".format(self.__class__.__name__))

        offspring = None
        if hasattr(individual, "constraints"):
            constraints = deepcopy(individual.constraints)
        else:
            constraints = []

        original_pbc = individual.pbc.copy()

        keep_offspring = False
        step = 0
        for step in range(self.max_retries):
            offspring = individual.copy()
            offspring.constraints = constraints
            offspring = self._perform_mutation(offspring)

            if offspring is None:
                keep_offspring = False
                continue

            offspring.wrap()

            offspring_is_valid = self.individual_is_valid_object(offspring)

            offspring_is_physical = self.individual_is_physical(offspring)

            if not offspring_is_valid:
                if self.logger:
                    self.logger.warning(
                        f"{self.__class__.__name__}: Offspring is not a valid object."
                        + f"\nOffspring: {offspring}"
                    )
                keep_offspring = False

            elif not offspring_is_physical:
                if self.logger:
                    self.logger.warning(
                        f"{self.__class__.__name__}: Offspring is not a physically feasible."
                        + f"\nOffspring: {offspring}"
                    )
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
            individual.set_pbc(original_pbc)

            if self.logger:
                self.logger.debug("Done! After {} steps.".format(step + 1))
            return individual, True

        else:
            if self.logger:
                self.logger.debug("Mutation failed.")
            return individual, False
