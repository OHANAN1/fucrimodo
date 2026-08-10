from .abstract import Mutation
from ....core import Individual
from ....core.utils.closest_distances_class import CustomClosestDistances
import ase_ga.standardmutations as ase_standard_mut
import numpy as np


class RattleMutation(Mutation):
    """
    Moves len(atoms) - n_top atoms in a random directions.
    The maximal movement is defined by max_movement.
    This is then limited by the closest distance between atoms.
    The closest distance is calculated by the
    ase_ga.utilities.closest_distances_generator.

    The bool shuffle_when_n_top determines if the atoms are shuffled
    if n_top is not "all". If False, always the first n_top atoms are moved.

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
        for i in indicees_to_rattle:
            positions[i] = positions[i] + self._rng.normal(
                scale=self.rattle_strength, size=positions[i].shape
            )

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
    Mirrors the individual along a random axis.
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        n_top: int | str = "all",
        max_retries: int = 100,
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
