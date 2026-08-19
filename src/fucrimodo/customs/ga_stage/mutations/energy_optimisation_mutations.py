from ase_ga import soft_mutation as ase_soft_mut

from ....core import Individual
from .abstract import Mutation


class SoftMutation(Mutation):
    """
    Apply a soft mutation to the individual using the ASE soft mutation.

    The mutation is deterministic, so ``max_retries`` is set to ``1``.
    The individual's ``confid`` info key is reset to ``0`` before applying
    the mutation.

    :param closest_distances: Minimum allowed interatomic distances used
        to validate the mutated structure.
    :type closest_distances: CustomClosestDistances
    :param max_retries: Maximum number of attempts to produce a valid
        mutation. Defaults to ``1``.
    :type max_retries: int
    :param rng: Random number generator passed to the base class.
        Defaults to ``None``.
    :type rng: None | np.random.Generator

    """

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
