from .abstract import Mutation
from ....core import Individual
from ....core.utils import CustomClosestDistances
from ase_ga import soft_mutation as ase_soft_mut


class SoftMutation(Mutation):
    def _perform_mutation(self, individual: Individual) -> Individual | None:
        # always keep at one, since this mutation works fully deterministic anyways
        self.max_retries = 1

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
