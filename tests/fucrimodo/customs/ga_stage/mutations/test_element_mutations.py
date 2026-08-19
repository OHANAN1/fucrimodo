from fucrimodo.customs.ga_stage.mutations.element_mutations import (
    PermutationMutation,
    ReplaceAtomsMutation,
)


class TestReplaceAtomsMutation:
    def test_mutate(
        self,
        closest_distances,
        cell_bounds,
        ind_crystal,
        mutation_reproducability_assessment,
    ):
        mut = ReplaceAtomsMutation(
            closest_distances=closest_distances,
            possible_elements=["H", "Na", "Cl", "O"],
        )
        mutation_reproducability_assessment(ind_crystal, mut, assess_change=True)


class TestPermutationMutation:
    def test_mutate(
        self,
        closest_distances,
        ind_molecule,
        mutation_reproducability_assessment,
    ):

        mut = PermutationMutation(
            closest_distances=closest_distances,
            prob=0.9,
            n_top=2,
        )
        mutation_reproducability_assessment(ind_molecule, mut, assess_change=False)
