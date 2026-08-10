from .abstract import Mutation
from ....core.utils import CustomClosestDistances
from ....core import Individual
from ...utils import LegacyRNGAdapter
import ase_ga.standardmutations as ase_standard_mut
import ase.data as ase_data
import numpy as np


class ReplaceAtomsMutation(Mutation):
    """
    Replace a random atom with an random atom species in the soap object.
    Atom has to be in the soap object species.

    Use soap_object.species to get the possible elements.
    Be aware! max_retries is counted as :arg:`max_retries_factor`. Max_retries will be factor*n_atoms in individual.
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
    """Uses ase atoms mutation PermutationMutation
    Uses :attr:`_legacy_rng` internally."""

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
