from .abstract import Mutation
from fucrimodo.core.modules import Individual
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from ase.ga import soft_mutation as ase_soft_mut
from ase.optimize import BFGS

# Implement with ML model from MACE here
# class RelaxationMutation(Mutation):
#     """
#     Relax the structure using the ASE BFGS optimizer.
#     """
#
#     def __init__(
#         self,
#         closest_distances: CustomClosestDistances,
#         max_steps_relaxation: int = 1000,
#         fmax: float = 0.05
#     ):
#         self.closest_distances = closest_distances
#         self.max_steps_relaxation = max_steps_relaxation
#         self.max_steps = 1
#         self.fmax = fmax
#
#     def __relax_crystal(
#         self,
#         crystal: Individual,
#         max_steps_relaxation: int,
#         fmax: float = 0.05,
#     ) -> Individual:
#         calc = EMT2()
#         crystal.calc = calc
#         dyn = BFGS(crystal, logfile="-")
#         dyn.run(fmax=fmax, steps=max_steps_relaxation)
#
#         return crystal
#
#     def perform_mutation(self, crystal: Individual) -> Individual | None:
#         offspring = self.__relax_crystal(
#             crystal, self.max_steps_relaxation, self.fmax
#         )
#         return offspring


class SoftMutation(Mutation):
    def __init__(
        self,
        closest_distances: CustomClosestDistances
    ) -> None:
        self.closest_distances = closest_distances
        self.max_steps = 1

    def perform_mutation(self, crystal: Individual) -> Individual | None:
        crystal.info["confid"] = 0
        if len(crystal.numbers) == 1:
            return None
        ase_soft = ase_soft_mut.SoftMutation(
            blmin=self.closest_distances,
            verbose=False,
            used_modes_file=None  # type: ignore
        )

        offspring = crystal
        mutant = ase_soft.mutate(offspring)
        return mutant
