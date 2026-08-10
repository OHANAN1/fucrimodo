from fucrimodo.customs.ga_stage.mutations.multi_mutation import MultipleMutations
from fucrimodo.customs.ga_stage.mutations.cell_mutations import (
    EnlargeMutation,
    RotationMutation,
)


class TestMultiMutation:
    def test_mutate(
        self,
        closest_distances,
        cell_bounds,
        ind_crystal,
        mutation_reproducability_assessment,
    ):

        mut = MultipleMutations(
            closest_distances=closest_distances,
            mutations=[
                EnlargeMutation(
                    closest_distances=closest_distances, cell_bounds=cell_bounds
                ),
                RotationMutation(closest_distances=closest_distances),
            ],
        )

        mutation_reproducability_assessment(ind_crystal, mut, assess_change=True)
