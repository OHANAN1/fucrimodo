from .abstract import Mutation
from fucrimodo.core.modules import Individual
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from ase.ga import soft_mutation as ase_soft_mut


class SoftMutation(Mutation):
    def __init__(self, closest_distances: CustomClosestDistances) -> None:
        self.closest_distances = closest_distances
        self.max_steps = 1

    def perform_mutation(self, individual: Individual) -> Individual | None:
        individual.info["confid"] = 0
        if len(individual.numbers) == 1:
            return None
        ase_soft = ase_soft_mut.SoftMutation(
            blmin=self.closest_distances,
            verbose=False,
            used_modes_file=None,  # type: ignore
        )

        offspring = individual
        mutant = ase_soft.mutate(offspring)
        return mutant
