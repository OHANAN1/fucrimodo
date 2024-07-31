from .abstract_mutation import Mutation
from ...utils.closest_distances_class import CustomClosestDistances
import ase
from ase.ga import soft_mutation as ase_soft_mut
from src.fucrimodo.utils.emt2 import EMT2
from ase.optimize import BFGS


class RelaxationMutation(Mutation):
    """
    Relax the structure using the ASE BFGS optimizer.
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        max_steps_relaxation: int = 1000,
        fmax: float = 0.05
    ):
        self.closest_distances = closest_distances
        self.max_steps_relaxation = max_steps_relaxation
        self.max_steps = 1
        self.fmax = fmax

    def __relax_crystal(
        self,
        crystal: ase.Atoms,
        max_steps_relaxation: int,
        fmax: float = 0.05,
    ) -> ase.Atoms:
        calc = EMT2()
        crystal.calc = calc
        dyn = BFGS(crystal, logfile="-")
        dyn.run(fmax=fmax, steps=max_steps_relaxation)

        return crystal

    def perform_mutation(self, crystal: ase.Atoms) -> ase.Atoms | None:
        offspring = self.__relax_crystal(
            crystal, self.max_steps_relaxation, self.fmax
        )
        return offspring


class SoftMutation(Mutation):
    def __init__(
        self,
        closest_distances: CustomClosestDistances
    ) -> None:
        self.closest_distances = closest_distances
        self.max_steps = 1

    def perform_mutation(self, crystal: ase.Atoms) -> ase.Atoms | None:
        crystal.info["confid"] = 0
        if len(crystal.numbers) == 1:
            return None
        ase_soft = ase_soft_mut.SoftMutation(
            blmin=self.closest_distances,
            verbose=True,
            used_modes_file=None  # type: ignore
        )

        offspring = crystal
        mutant = ase_soft.mutate(offspring)
        return mutant
