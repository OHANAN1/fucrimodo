from .abstract_mutation import Mutation
from ...utils.closest_distances_class import CustomClosestDistances
import ase.ga.standardmutations as ase_standard_mut
import ase
import numpy as np

class RattleMutation(Mutation):
    """
    Moves n_max atoms randomly in all 3 directions.
    The maximal movement is defined by max_movement.
    This is then limited by the closest distance between atoms.
    The closest distance is calculated by the
    ase.ga.utilities.closest_distances_generator.

    The bool shuffle_when_n_top determines if the atoms are shuffled
    if n_top is not "all". If False, always the first n_top atoms are moved.

    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        n_top: int | str = "all",
        rattle_strength: float = 0.8,
        rattle_prop: float = 0.5,
        shuffle_when_n_top: bool = True
    ) -> None:
        self.n_top = n_top
        self.rattle_strength = rattle_strength
        self.rattle_prop = rattle_prop
        self.closest_distances = closest_distances
        self.max_steps = 1
        self.shuffle_when_n_top = shuffle_when_n_top

        if not isinstance(n_top, int) and not n_top == "all":
            raise ValueError(
                "n_top has to be an integer or the string 'all'"
            )

    def perform_mutation(self, crystal: ase.Atoms) -> ase.Atoms | None:
        if self.n_top == "all":
            n_top = len(crystal)
        else:
            n_top = self.n_top

            if self.shuffle_when_n_top:
                shuffled_crystal = crystal[
                    np.random.permutation(len(crystal))
                ]
                if isinstance(shuffled_crystal, ase.Atoms):
                    crystal = shuffled_crystal
                else:
                    raise ValueError("Shuffling did not work")

        ase_rattle = ase_standard_mut.RattleMutation(
            n_top=n_top,
            blmin=self.closest_distances,
            rattle_strength=self.rattle_strength,
            rattle_prop=self.rattle_prop,
            verbose=False
        )

        offspring = crystal
        mutant = ase_rattle.mutate(offspring)

        return mutant


class MirrorMutation(Mutation):
    """
    Mirrors the crystal along a random axis.
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        n_top: int | str = "all",
    ) -> None:
        self.closest_distances = closest_distances
        self.max_steps = 1
        self.n_top = n_top

        if not isinstance(n_top, int) and not n_top == "all":
            raise ValueError(
                "n_top has to be an integer or the string 'all'"
            )

    def perform_mutation(self, crystal: ase.Atoms) -> ase.Atoms | None:

        if self.n_top == "all":
            n_top = len(crystal)
        else:
            n_top = self.n_top

        ase_mirror = ase_standard_mut.MirrorMutation(
            blmin=self.closest_distances,
            n_top=n_top,
            verbose=False
        )

        offspring = crystal
        mutant = ase_mirror.mutate(offspring)

        return mutant
