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
from ..utils import LegacyRNGAdapter

# ╔══════════════════════════════════════════════════════════╗
# ║                 Abstract Crossover Class                 ║
# ╚══════════════════════════════════════════════════════════╝


class Crossover(ABC):
    """Abstract base class for all crossover operators.

    This class defines the common interface and shared utilities used by
    every concrete crossover implementation. It handles retry logic,
    validation of offspring objects, and restoration of periodic
    boundary conditions.

    A subclass only needs to implement :meth:`_perform_crossover`. The
    public :meth:`crossover` method then repeatedly calls it, checks the
    resulting offspring, and returns the final result together with a
    success flag.

    .. note::
        Subclasses must call ``super().__init__(...)`` so that
        :attr:`_rng`, :attr:`max_retries`, and :attr:`closest_distances`
        are initialized correctly.

    :param closest_distances: Validator used to check whether atoms in an
        offspring are unphysically close.
    :param max_retries: Number of times a failed crossover attempt is
        retried before giving up. Defaults to ``1``.
    :param rng: Random number generator used for stochastic operations.
        If ``None``, ``np.random.default_rng()`` is used.

    :ivar _rng: Random number generator available to subclasses.
    :ivar closest_distances: Closest-distance validator.
    :ivar max_retries: Maximum number of crossover retries.
    :ivar required_steps: Number of attempts used in the last call to
        :meth:`crossover`. Set after each call.
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

    def __repr__(self):
        """Return a compact string representation of the operator.

        Excludes ``closest_distances`` and ``cell_bounds`` from the output.
        """
        class_name = self.__class__.__name__
        variables = vars(self)

        variables_str = " "
        for key, value in variables.items():
            if key == "closest_distances" or key == "cell_bounds":
                continue

            variables_str += f"{key}={value}, "

        variables_str = variables_str[:-2]
        return f"{class_name}({variables_str})"

    @property
    def logger(self) -> logging.Logger | None:
        """Optional logger for debug and warning messages.

        :return: The logger if one has been set, otherwise ``None``.
        :rtype: logging.Logger | None
        """
        if not hasattr(self, "_logger"):
            return None
        return self._logger

    @logger.setter
    def logger(self, value):
        """Set the logger used by this crossover operator."""
        self._logger = value

    def _individual_is_valid_object(self, individual: Individual) -> bool:
        """Check whether ``individual`` is a usable :class:`Individual`.

        An individual is considered valid when it is an :class:`Individual`
        instance, contains at least one :class:`ase.Atom`, and none of its
        positions, cell vectors, or atomic numbers are ``NaN``.

        :param individual: Structure to validate.
        :return: ``True`` if the individual is valid, ``False`` otherwise.
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
        """Check whether ``individual`` satisfies basic physical constraints.

        Returns ``False`` if the cell volume is smaller than ``1.0`` (chosen
        based of experience) or if any pair of atoms is closer than the allowed
        distances defined by :attr:`closest_distances`.

        :param individual: Structure to validate.
        :type individual: Individual
        :return: ``True`` if the individual is physical, ``False`` otherwise.
        :rtype: bool
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
        """Perform the actual crossover operation.

        This method receives copies of the two parents and must return two
        offspring individuals. Returning ``None`` for either offspring is
        interpreted as a failed crossover attempt.

        No validity or physical checks are performed here; those are handled
        by :meth:`crossover`.

        :param parent1: Copy of the first parent.
        :type parent1: Individual
        :param parent2: Copy of the second parent.
        :type parent2: Individual
        :return: Two offspring individuals, or ``(None, None)`` on failure.
        :rtype: tuple[Individual, Individual] | tuple[None, None]
        """
        pass

    def crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual, bool]:
        """Generate two offspring from ``parent1`` and ``parent2``.

        The crossover is attempted up to :attr:`max_retries` times. Each
        attempt creates fresh copies of the parents, applies
        :meth:`_perform_crossover`, and validates the resulting offspring.
        If a valid and physical pair is produced, it is returned with
        ``True``. Otherwise, copies of the original parents are returned
        with ``False``.

        .. note::
            The returned individuals are always copies and reset. Parents remain
            untouched.

        (`Respect your father!` ~ Fujimoto)

        :param parent1: First parent individual.
        :type parent1: Individual
        :param parent2: Second parent individual.
        :type parent2: Individual

        :return: A tuple containing the two offspring and a success flag.
        :rtype: tuple[Individual, Individual, bool]
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
            offspring_1.reset()
            offspring_2 = parent2.copy()
            offspring_2.reset()

            offspring_1, offspring_2 = self._perform_crossover(offspring_1, offspring_2)

            if offspring_1 is None or offspring_2 is None:
                keep_offspring = False
                continue

            offspring_1.set_pbc(par1_pbc)
            offspring_2.set_pbc(par2_pbc)

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
            break

        self.required_steps = step

        if keep_offspring and offspring_1 and offspring_2:
            if self.logger:
                self.logger.debug("Done! After {} steps.".format(step + 1))
            return (offspring_1, offspring_2, True)
        else:
            # Generate copies of parents and reset them
            offspring_1 = parent1.copy()
            offspring_1.reset()
            offspring_2 = parent2.copy()
            offspring_2.reset()
            if self.logger:
                self.logger.debug("Crossover failed.")
            return (offspring_1, offspring_2, False)


# ╔══════════════════════════════════════════════════════════╗
# ║                    Crossover Classes                     ║
# ╚══════════════════════════════════════════════════════════╝


class UnitCellCrossover(Crossover):
    """
    Crossover operator that mixes unit-cell vectors between two parents.

    For each of the three cell vectors, one vector is randomly drawn from
    either parent and combined into two new unit cells. Atomic positions
    can be scaled together with the new cell via ``scale_atoms``.

    :param closest_distances: Closest-distance constraints used by the
        base crossover class.
    :type closest_distances: CustomClosestDistances
    :param scale_atoms: If ``True``, scale atomic positions when applying
        the new cell. Defaults to ``True``.
    :type scale_atoms: bool
    :param max_retries: Maximum number of retries allowed by the base
        crossover class. Defaults to ``10``.
    :type max_retries: int
    :param rng: Random number generator used to shuffle cell vectors. If
        ``None``, the base class handles default initialization.
    :type rng: None | np.random.Generator
    """

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
    """Crossover operator that stacks two parent cells along a random axis.

    The selected cell vector is replaced by the sum of the corresponding
    vectors from both parents, producing a combined cell. The resulting
    cell is checked against ``cell_bounds`` before the atomic structures
    are stacked along the same axis.

    :param closest_distances: Closest-distance constraints used by the
        base crossover class.
    :type closest_distances: CustomClosestDistances
    :param cell_bounds: Bounds used to validate the stacked cell.
    :type cell_bounds: CustomCellBounds
    :param scale_atoms: If ``True``, scale atomic positions when applying
        the new cell. Defaults to ``True``.
    :type scale_atoms: bool
    :param max_retries: Maximum number of retries allowed by the base
        crossover class. Defaults to ``10``.
    :type max_retries: int
    :param rng: Random number generator used to select the stacking axis.
        If ``None``, the base class handles default initialization.
    :type rng: None | np.random.Generator
    """

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

        axis = self._rng.integers(0, 3)

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
    """One-point crossover operator that exchanges atomic numbers between parents.

    A random cut index is chosen along the atom list. The offspring inherit
    atomic numbers from one parent up to the cut and from the other parent
    beyond it. Positions are kept from the respective parent copy and wrapped.

    Crossover is aborted if either parent has fewer than two atoms or if both
    parents contain only a single element type.

    :param closest_distances: Closest-distance constraints used by the
        base crossover class.
    :type closest_distances: CustomClosestDistances
    :param max_retries: Maximum number of retries allowed by the base
        crossover class. Defaults to ``10``.
    :type max_retries: int
    :param rng: Random number generator used to select the stacking axis.
        If ``None``, the base class handles default initialization.
    :type rng: None | np.random.Generator
    """

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
    """One-point crossover operator that exchanges atomic positions between parents.

    A random cut index is chosen along the atom list. Each offspring keeps the
    atomic numbers of one parent copy, but receives positions from the other
    parent before the cut and from its own parent after the cut. Positions are
    wrapped into the cell afterwards.

    Crossover is aborted if either parent has fewer than two atoms.

    :param closest_distances: Closest-distance constraints used by the
        base crossover class.
    :type closest_distances: CustomClosestDistances
    :param max_retries: Maximum number of retries allowed by the base
        crossover class. Defaults to ``10``.
    :type max_retries: int
    :param rng: Random number generator used to select the stacking axis.
        If ``None``, the base class handles default initialization.
    :type rng: None | np.random.Generator

    """

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
    """
    Cut-and-splice crossover operator based on ASE_GA's CutAndSplicePairing.

    This operator cuts two parent structures along a random plane and
    splices their top parts together to create two offspring. Parents are
    sorted by atomic number and must share the same stoichiometry, the
    same number of atoms, and contain at least two atoms. Non-variable
    cell vectors must match between the two parents.

    More info in `ase_ga repo <https://github.com/dtu-energy/ase-ga>`__.

    :param closest_distances: Closest-distance constraints used by the
        pairing operator and the base crossover class.
    :type closest_distances: CustomClosestDistances
    :param cell_bounds: Bounds used to validate the generated offspring
        cell.
    :type cell_bounds: CustomCellBounds
    :param n_top: Number of top atoms to include in the crossover, or
        ``"all"`` to use all atoms. Defaults to ``"all"``.
    :type n_top: int | str
    :param number_of_variable_cell_vectors: Number of cell vectors that
        are allowed to differ between parents. The remaining vectors must
        match. Defaults to ``0``.
    :type number_of_variable_cell_vectors: int
    :param max_retries: Maximum number of retries allowed by the base
        crossover class. Defaults to ``10``.
    :type max_retries: int
    :param rng: Random number generator used by the pairing operator. If
        ``None``, the base class handles default initialization.
    :type rng: None | np.random.Generator
    """

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
            Individual.from_ase(offspring_1),
            Individual.from_ase(offspring_2),
        )
