from .abstract import Mutation
from fucrimodo.core.modules import Individual
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
import random
import logging
logger = logging.getLogger('run_logger')


class MultipleMutations(Mutation):
    def __init__(
        self,
        mutations: list[Mutation],
        closest_distances: CustomClosestDistances,
        number_of_mutations: int | str = "all",
        random_order: bool = True,
        can_occure_multiple_times: bool = False
    ) -> None:
        self.random_order = random_order
        self.mutations = mutations
        self.closest_distances = closest_distances
        self.can_occure_multiple_times = can_occure_multiple_times
        self.max_steps = 1

        if (
            isinstance(number_of_mutations, str)
                and number_of_mutations == "all"
        ):
            self.number_of_mutations: int = len(self.mutations)
        elif isinstance(number_of_mutations, int):
            self.number_of_mutations: int = int(number_of_mutations)
        else:
            raise ValueError(
                "number_of_mutations has to be an integer or the string 'all'"
            )

    def __get_max_steps_from_mutations(self, mutations: list[Mutation]) -> int:
        max_steps = []
        for mutation in mutations:
            if hasattr(mutation, "max_steps"):
                max_steps.append(mutation.max_steps)
            else:
                max_steps.append(1)

        max_steps = min(max_steps)

        return max_steps

    def __select_mutations(self) -> list[Mutation]:
        if self.random_order:
            random.shuffle(self.mutations)

        if not self.can_occure_multiple_times:
            selected_mutations = self.mutations[:self.number_of_mutations]
        else:
            selected_mutations = random.choices(
                self.mutations, k=self.number_of_mutations
            )

        return selected_mutations

    def perform_mutation(self, crystal: Individual) -> Individual | None:
        offspring = crystal

        selected_mutations = self.__select_mutations()

        self.max_steps = self.__get_max_steps_from_mutations(
            selected_mutations
        )

        for i in range(self.number_of_mutations):
            mutation = selected_mutations[i]
            logger.info(f"\tPerforming {mutation.__class__.__name__}")
            offspring = self.mutations[i].perform_mutation(offspring)

            if offspring is None:
                return None

        return offspring
