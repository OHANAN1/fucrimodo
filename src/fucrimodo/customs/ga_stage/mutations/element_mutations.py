import ase.data as ase_data
import ase_ga.standardmutations as ase_standard_mut
import numpy as np

from ....core import Individual
from ....core.utils import CustomClosestDistances
from ...utils import LegacyRNGAdapter
from .abstract import Mutation


class ReplaceAtomsMutation(Mutation):
    """
    Replace atoms in an individual with random species from a given pool.

    For each atom to replace, a random atom is selected and its atomic
    number is changed to a different species from ``possible_elements``.
    The replacement is applied in place.

    :param possible_elements: List of element symbols that may be used as
        replacement species.
    :type possible_elements: list[str]
    :param closest_distances: Minimum allowed interatomic distances used
        to validate the mutated structure.
    :type closest_distances: CustomClosestDistances
    :param n_atoms_to_replace: Number of atoms to replace per mutation.
        Defaults to ``1``.
    :type n_atoms_to_replace: int
    :param max_retries: Factor used to compute the actual number of retries
        as ``max_retries * n_atoms``. Defaults to ``2``.
    :type max_retries: int
    :param rng: Random number generator. If ``None``, the base class
        creates one.
    :type rng: None | np.random.Generator
    """

    # NOTE: Maybe use the ase RandomElementMutation
    def __init__(
        self,
        possible_elements: list["str"],
        closest_distances: CustomClosestDistances,
        n_atoms_to_replace: int = 1,
        max_retries: int = 2,
        rng: None | np.random.Generator = None,
    ):
        super().__init__(
            closest_distances=closest_distances, max_retries=max_retries, rng=rng
        )
        self.max_retries_factor = max_retries

        self.possible_atomic_numbers: list[int] = [
            ase_data.atomic_numbers[atom] for atom in possible_elements
        ]
        self.n_atoms_to_replace = n_atoms_to_replace

    def _perform_mutation(
        self,
        individual: Individual,
    ) -> Individual | None:
        self.max_retries = self.max_retries_factor * len(individual)

        if len(individual) < self.n_atoms_to_replace:
            return None

        atomic_numbers = individual.get_atomic_numbers()

        for _ in range(self.n_atoms_to_replace):
            random_index = self._rng.choice(range(len(atomic_numbers)))
            selected_atomic_number = atomic_numbers[random_index]
            remaining_atomic_numbers = [
                number
                for number in self.possible_atomic_numbers
                if number != selected_atomic_number
            ]
            if len(remaining_atomic_numbers) == 0:
                return None
            else:
                atomic_numbers[random_index] = self._rng.choice(
                    remaining_atomic_numbers
                )

        individual.set_atomic_numbers(atomic_numbers)
        return individual


class PermutationMutation(Mutation):
    """
    Swap positions of atoms with different species.

    This wraps the ASE ``PermutationMutation`` and uses a legacy RNG
    adapter internally to provide the required random number generator
    interface.

    :param closest_distances: Minimum allowed interatomic distances used
        to validate the mutated structure.
    :type closest_distances: CustomClosestDistances
    :param n_top: Number of atoms from the top of the structure to consider
        for permutation, or the string ``"all"`` to use all atoms. Defaults
        to ``"all"``.
    :type n_top: int | str
    :param prob: Probability of attempting a permutation. Defaults to
        ``0.5``.
    :type prob: float
    :param max_retries: Maximum number of attempts to produce a valid
        mutation. Defaults to ``1``.
    :type max_retries: int
    :param rng: Random number generator. If ``None``, the base class
        creates one.
    :type rng: None | np.random.Generator
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        n_top: int | str = "all",
        prob: float = 0.5,
        max_retries: int = 1,
        rng: None | np.random.Generator = None,
    ):
        super().__init__(
            closest_distances=closest_distances, max_retries=max_retries, rng=rng
        )
        self._legacy_rng = LegacyRNGAdapter(self._rng)

        self.n_top = n_top
        self.prob = prob

        if not isinstance(n_top, int) and not n_top == "all":
            raise ValueError("n_top has to be an integer or the string 'all'")

        # Define the ase mutation.
        self.ase_permutation = ase_standard_mut.PermutationMutation(
            n_top=n_top,
            probability=self.prob,
            blmin=self.closest_distances,
            verbose=True,
            rng=self._legacy_rng,  # type: ignore
        )

    def _perform_mutation(self, individual: Individual) -> Individual | None:
        if len(individual.numbers) == 1:
            return None
        if len(np.unique(individual.numbers)) == 1:
            return None

        # The n_top must be adjusted at every step if its all
        # TODO: Check if the ase mutation does that also
        if self.n_top == "all":
            n_top = len(individual)
        else:
            n_top = self.n_top
        self.ase_permutation.n_top = n_top

        offspring = individual
        mutant = self.ase_permutation.mutate(offspring)
        return mutant
