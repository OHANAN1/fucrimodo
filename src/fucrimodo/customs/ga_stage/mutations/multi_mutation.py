import numpy as np

from ....core import Individual
from ....core.utils import CustomClosestDistances
from ...utils import LegacyRNGAdapter

from .abstract import Mutation


class MultipleMutations(Mutation):
    """Uses :attr:`_legacy_rng` internally. But changing :attr:`_rng` changes both!

    ensures that the mutations use the same rng automatically
    """

    def __init__(
        self,
        mutations: list[Mutation],
        closest_distances: CustomClosestDistances,
        number_of_mutations: int | str = "all",
        random_order: bool = True,
        can_occure_multiple_times: bool = False,
        max_retries: int = 100,
        rng: None | np.random.Generator = None,
    ):
        super().__init__(
            closest_distances=closest_distances, max_retries=max_retries, rng=rng
        )

        self.random_order = random_order
        self.mutations = mutations
        self.closest_distances = closest_distances
        self.can_occure_multiple_times = can_occure_multiple_times
        self.max_retries = 1

        if isinstance(number_of_mutations, str) and number_of_mutations == "all":
            self.number_of_mutations: int = len(self.mutations)
        elif isinstance(number_of_mutations, int):
            self.number_of_mutations: int = int(number_of_mutations)
        else:
            raise ValueError(
                "number_of_mutations has to be an integer or the string 'all'"
            )

    def __get_max_steps_from_mutations(self, mutations: list[Mutation]) -> int:
        max_retries = []
        for mutation in mutations:
            if hasattr(mutation, "max_retries"):
                max_retries.append(mutation.max_retries)
            else:
                max_retries.append(1)

        max_retries = min(max_retries)

        return max_retries

    def __select_mutations(self) -> list[Mutation]:
        if self.random_order:
            mutation_order = self._legacy_rng.permutation(self.mutations).tolist()
        else:
            mutation_order = list(self.mutations)

        if not self.can_occure_multiple_times:
            return mutation_order[: self.number_of_mutations]
        else:
            return self._legacy_rng.choices(mutation_order, k=self.number_of_mutations)

    def _perform_mutation(self, individual: Individual) -> Individual | None:
        offspring = individual

        # Update the rng of the mutations, for the chase it was changed
        for mut in self.mutations:
            mut._rng = self._rng
            if hasattr(mut, "_legacy_rng"):
                mut._legacy_rng = self._legacy_rng  # type: ignore

        selected_mutations = self.__select_mutations()

        self.max_retries = self.__get_max_steps_from_mutations(selected_mutations)

        for i in range(self.number_of_mutations):
            mutation = selected_mutations[i]
            if self.logger:
                self.logger.debug(f"\tPerforming {mutation.__class__.__name__}")
            offspring = mutation._perform_mutation(individual=offspring)

            if offspring is None:
                return None

        return offspring
