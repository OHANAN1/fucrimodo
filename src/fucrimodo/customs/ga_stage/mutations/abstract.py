import logging
from abc import ABC, abstractmethod
from copy import deepcopy

import ase
import numpy as np

from fucrimodo.core import Individual
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances

from ...utils import LegacyRNGAdapter

# ╔══════════════════════════════════════════════════════════╗
# ║            Abstract Base Class for Mutations             ║
# ╚══════════════════════════════════════════════════════════╝


class Mutation(ABC):
    """Abstract base class for all mutations acting on an ASE-based ``Individual``.

    The public entry point :meth:`mutate` handles retry logic, validation, and
    in-place replacement of the parent when a valid and physically feasible
    offspring is produced.

    .. note::
        Subclasses must call ``super().__init__(...)`` so that
        :attr:`_rng`, :attr:`_legacy_rng`, :attr:`max_retries`, and :attr:`closest_distances`
        are initialized correctly.

    :param closest_distances: Object that checks whether atoms in an individual
        are too close to each other.
    :param max_retries: Maximum number of attempts to create a valid offspring.
        Defaults to ``1``.
    :param rng: NumPy random number generator. If ``None``, a new default
        generator is created.

    :ivar closest_distances: Distance checker used to validate physical
        feasibility of offspring.
    :vartype closest_distances: CustomClosestDistances
    :ivar max_retries: Maximum number of mutation attempts.
    :vartype max_retries: int
    :ivar _rng: Internal NumPy random number generator.
    :vartype _rng: numpy.random.Generator
    :ivar _legacy_rng: Legacy-style RNG adapter wrapping :attr:`_rng`.
    :vartype _legacy_rng: fucrimodo.customs.utils.LegacyRNGAdapter
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        max_retries: int = 1,
        rng: None | np.random.Generator = None,
        deterministic: bool = False,
    ):
        if not rng:
            rng = np.random.default_rng()
        self._rng = rng
        self._legacy_rng = LegacyRNGAdapter(self._rng)

        self.closest_distances = closest_distances
        self.max_retries = max_retries

    @property
    def logger(self) -> logging.Logger | None:
        """
        Logger used by this mutation instance.

        :return: The logger, or ``None`` if none was assigned.
        """
        if not hasattr(self, "_logger"):
            return None
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    @property
    def rng(self) -> np.random.Generator:
        """
        NumPy random number generator used by this mutation instance.

        :return: The random number generator.
        """
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

    def _individual_is_valid_object(self, individual: Individual) -> bool:
        """
        Check whether ``individual`` is a well-formed ``Individual`` object.

        Checks that the object is an ``Individual``, is non-empty, contains only
        ``ase.Atom`` objects, and has no NaN values in positions, cell, or atomic
        numbers.

        :param individual: The individual to validate.
        :return: ``True`` if the individual is a valid object, ``False`` otherwise.
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
        Check whether ``individual`` represents a physically feasible structure.

        Rejects individuals whose cell volume is below ``1.0`` (if a cell exists)
        and individuals whose atoms are too close according to
        ``closest_distances``.

        :param individual: The individual to check.
        :return: ``True`` if the individual is physical, ``False`` otherwise.
        """
        if not np.all(individual.cell == 0):
            if individual.get_volume() < 1.0:
                return False

        if self.closest_distances.atoms_are_too_close(individual):
            return False

        return True

    @abstractmethod
    def _perform_mutation(self, individual: Individual) -> Individual | None:
        """
        Apply the mutation to ``individual`` and return the offspring.

        Perform the actual mutation operation.

        This method receives one individual and modify it.
        Returning ``None`` for either the individual is
        interpreted as a failed mutation attempt.

        No validity or physical checks are performed here; those are handled
        by :meth:`mutate`.

        :param individual: The parent individual to mutate. It is already a copy
            owned by :meth:`mutate`.
        :return: The mutated offspring, or ``None`` if the mutation could not be
            performed.
        """
        pass

    def mutate(self, individual: Individual) -> tuple[Individual, bool]:
        r"""
        Try to mutate ``individual`` up to ``max_retries`` times.

        On success, the parent ``individual`` is updated in place with the offspring
        atoms, cell, constraints, and original periodic boundary conditions, and
        the method returns the updated individual and ``True``. On failure, the
        original individual and ``False`` are returned.

        .. note::
            The returned individuals links to the original individual.

        :param individual: The parent individual to mutate.

        :return: A tuple of ``(individual, success)`` where ``success`` is ``True``
            if a valid and physical offspring was produced.

        .. code-block:: text

                            O  o
                       _\_   o
             >('>   \\/  o\ .
                    //\___=
                       ''

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

            offspring_is_valid = self._individual_is_valid_object(offspring)
            if not offspring_is_valid:
                if self.logger:
                    self.logger.warn(
                        f"{self.__class__.__name__}: Offspring is not a valid object."
                        + f"\nOffspring: {offspring}"
                    )
                keep_offspring = False
                continue

            offspring.wrap()

            offspring_is_physical = self._individual_is_physical(offspring)
            if not offspring_is_physical:
                if self.logger:
                    self.logger.debug(
                        f"{self.__class__.__name__}: Offspring is not a physically feasible."
                        + f"\nOffspring: {offspring}"
                    )
                keep_offspring = False
                continue

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
