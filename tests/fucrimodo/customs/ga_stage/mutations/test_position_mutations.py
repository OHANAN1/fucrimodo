from fucrimodo.customs.ga_stage.mutations.position_mutations import (
    RattleMutation,
    MirrorMutation,
)


class TestRattleMutation:
    def test_mutate(
        self,
        closest_distances,
        ind_crystal,
        ind_molecule,
        ind_slab,
        mutation_reproducability_assessment,
    ):

        mut = RattleMutation(
            n_top=2,
            closest_distances=closest_distances,
            rattle_strength=0.01,
            rattle_prop=1.0,
        )

        mutation_reproducability_assessment(ind_slab, mut, assess_change=True)
        mutation_reproducability_assessment(ind_molecule, mut, assess_change=True)
        mutation_reproducability_assessment(ind_crystal, mut, assess_change=True)


class TestMirrorMutation:
    def test_mutate(
        self,
        closest_distances,
        ind_crystal,
        mutation_reproducability_assessment,
    ):

        mut = MirrorMutation(
            n_top=2,
            closest_distances=closest_distances,
            max_retries=100,
        )

        mutation_reproducability_assessment(ind_crystal, mut, assess_change=True)
