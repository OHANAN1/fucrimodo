import numpy as np

from ....core import Individual
from ....core.utils import CustomClosestDistances
from .abstract import Mutation


class MultipleMutations(Mutation):
    """
    Apply multiple mutations sequentially to an individual.

    A subset of the provided mutations is selected and applied in order.
    The selection order is randomized when ``random_order`` is ``True``.
    Mutations are sampled without replacement by default, meaning each
    mutation is used at most once per application.

    The RNG of each sub-mutation is synchronized with this object's RNG
    before every application, so all mutations share the same random state.

    :param mutations: List of mutations to apply.
    :type mutations: list[Mutation]
    :param closest_distances: Minimum allowed interatomic distances used
        for validation.
    :type closest_distances: CustomClosestDistances
    :param number_of_mutations: Number of mutations to apply, or the string
        ``"all"`` to apply all of them. Defaults to ``"all"``.
    :type number_of_mutations: int | str
    :param random_order: Whether to randomize the order of mutations
        before selection. Defaults to ``True``.
    :type random_order: bool
    :param can_occure_multiple_times: Whether the same mutation may be
        selected multiple times. Defaults to ``False``.
    :type can_occure_multiple_times: bool
    :param max_retries: Maximum number of attempts to produce a valid
        mutation. Defaults to ``100``.
    :type max_retries: int
    :param rng: Random number generator. If ``None``, the base class
        creates one.
    :type rng: None | np.random.Generator
    """

    def __init__(
        self,
        mutations: list[Mutation],
        closest_distances: CustomClosestDistances,
        number_of_mutations: int | str = "all",
        random_order: bool = True,
        can_occure_multiple_times: bool = False,
        max_retries: int = 10,
        rng: None | np.random.Generator = None,
    ):
        super().__init__(
            closest_distances=closest_distances, max_retries=max_retries, rng=rng
        )

        self.random_order = random_order
        self.mutations = mutations
        self.closest_distances = closest_distances
        self.can_occure_multiple_times = can_occure_multiple_times
        self.max_retries = max_retries

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
