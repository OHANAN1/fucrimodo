import ase_ga.standardmutations as ase_standard_mut
import numpy as np

from ....core import Individual
from ....core.utils.closest_distances_class import CustomClosestDistances
from .abstract import Mutation


class RattleMutation(Mutation):
    """
    Randomly displace a subset of atoms.

    ``n_top`` atoms are selected without replacement and their positions
    are perturbed by a normally distributed displacement with standard
    deviation ``rattle_strength``. The displacement is applied through
    ``set_positions``, so any ASE constraints on the individual are
    respected.

    :param closest_distances: Minimum allowed interatomic distances used
        to validate the mutated structure.
    :type closest_distances: CustomClosestDistances
    :param n_top: Number of atoms to rattle, or the string ``"all"`` to
        rattle all atoms. Defaults to ``"all"``.
    :type n_top: int | str
    :param rattle_strength: Standard deviation of the Gaussian displacement
        in Angstrom. Defaults to ``0.8``.
    :type rattle_strength: float
    :param rattle_prop: Probability that a selected atom is actually rattled.
        Defaults to ``0.5``.
    :type rattle_prop: float
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
        n_top: int | str = "all",
        rattle_strength: float = 0.8,
        rattle_prop: float = 0.5,
        max_retries: int = 100,
        rng: None | np.random.Generator = None,
    ):
        super().__init__(
            closest_distances=closest_distances, max_retries=max_retries, rng=rng
        )
        self.n_top = n_top
        self.rattle_strength = rattle_strength
        self.rattle_prop = rattle_prop
        self.closest_distances = closest_distances

        if not isinstance(n_top, int) and not n_top == "all":
            raise ValueError("n_top has to be an integer or the string 'all'")

    def _perform_mutation(self, individual: Individual) -> Individual | None:
        # Set n_top to the desired value
        if self.n_top == "all":
            n_top = len(individual)
        else:
            n_top = int(self.n_top)

        # Make sure n_top is not larger than the number of atoms
        n_top = min(n_top, len(individual))

        positions = individual.get_positions().copy()
        indicees_to_rattle = self._rng.choice(len(individual), n_top, replace=False)

        mutated = False
        for i in indicees_to_rattle:
            if self._rng.random() < self.rattle_prop:
                positions[i] = positions[i] + self._rng.normal(
                    scale=self.rattle_strength, size=positions[i].shape
                )
                mutated = True

        # If no atom was rattled, the mutation did nothing
        if not mutated:
            return None

        # Set the positions of the individual
        # This also respects the constraints set
        individual.set_positions(positions)

        # If constrains of Atoms made rattle movement not applicable
        # return None to signalize mutation failed
        if np.any(positions != individual.get_positions()):
            return None

        return individual


class MirrorMutation(Mutation):
    """
    Mirror the individual along a random axis.

    This wraps the ASE ``MirrorMutation``. The mutation is applied to the
    top ``n_top`` atoms of the structure.

    :param closest_distances: Minimum allowed interatomic distances used
        to validate the mutated structure.
    :type closest_distances: CustomClosestDistances
    :param n_top: Number of atoms from the top of the structure to consider
        for mirroring, or the string ``"all"`` to use all atoms. Defaults
        to ``"all"``.
    :type n_top: int | str
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
        n_top: int | str = "all",
        max_retries: int = 1,
        rng: None | np.random.Generator = None,
    ):
        super().__init__(
            closest_distances=closest_distances, max_retries=max_retries, rng=rng
        )

        self.closest_distances = closest_distances
        self.max_steps = 1
        self.n_top = n_top

        if not isinstance(n_top, int) and not n_top == "all":
            raise ValueError("n_top has to be an integer or the string 'all'")

    def _perform_mutation(self, individual: Individual) -> Individual | None:

        if self.n_top == "all":
            n_top = len(individual)
        else:
            n_top = self.n_top

        ase_mirror = ase_standard_mut.MirrorMutation(
            blmin=self.closest_distances,
            n_top=n_top,
            verbose=False,
            rng=self._legacy_rng,  # type: ignore
        )

        offspring = individual
        mutant = ase_mirror.mutate(offspring)

        return mutant
