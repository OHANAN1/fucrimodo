from fucrimodo.core import Individual
from fucrimodo.customs.ga_stage.mutations.energy_optimisation_mutations import (
    SoftMutation,
)


class TestSoftMutation:
    # NOTE: No extencive test for deterministic behaviour needed, bc
    def test_mutate(
        self,
        closest_distances,
        ind_crystal,
        ind_molecule,
        mutation_reproducability_assessment,
    ):
        mut = SoftMutation(closest_distances=closest_distances)

        mutation_reproducability_assessment(ind_crystal, mut, assess_change=False)

        new_ind, success = mut.mutate(ind_crystal)
        assert success
        assert type(new_ind) == Individual

        new_ind, success = mut.mutate(ind_molecule)
        assert success
