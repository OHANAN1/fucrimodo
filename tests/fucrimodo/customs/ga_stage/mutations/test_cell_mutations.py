from fucrimodo.customs.ga_stage.mutations.cell_mutations import (
    EnlargeMutation,
    MinimizeTiltMutation,
    NiggliReduceMutation,
    RotationMutation,
    ScaleUnitCellMutation,
    StrainMutation,
)


class TestScaleUnitCellMutation:
    def test_mutate(
        self,
        closest_distances,
        cell_bounds,
        ind_crystal,
        mutation_reproducability_assessment,
    ):
        mut = ScaleUnitCellMutation(
            closest_distances=closest_distances,
            cell_bounds=cell_bounds,
        )
        mutation_reproducability_assessment(ind_crystal, mut, assess_change=True)


class TestStrainMutation:
    def test_mutate(
        self,
        closest_distances,
        cell_bounds,
        ind_crystal,
        mutation_reproducability_assessment,
    ):
        mut = StrainMutation(
            closest_distances=closest_distances,
            cell_bounds=cell_bounds,
            n_variable_cell_vectors=3,
            stddev=0.7,
            max_retries=100,
        )
        mutation_reproducability_assessment(ind_crystal, mut, assess_change=True)


class TestEnlargeMutation:
    def test_mutate(
        self,
        closest_distances,
        cell_bounds,
        ind_crystal,
        mutation_reproducability_assessment,
    ):
        mut = EnlargeMutation(
            closest_distances=closest_distances,
            cell_bounds=cell_bounds,
            max_retries=100,
        )
        mutation_reproducability_assessment(ind_crystal, mut, assess_change=False)


class TestNiggliReduceMutation:
    def test_mutate(
        self,
        closest_distances,
        ind_crystal,
        mutation_reproducability_assessment,
    ):
        mut = NiggliReduceMutation(
            closest_distances=closest_distances,
            max_retries=1,
        )
        mutation_reproducability_assessment(ind_crystal, mut, assess_change=False)


class TestMinimizeTiltMutation:
    def test_mutate(
        self,
        closest_distances,
        ind_crystal,
        mutation_reproducability_assessment,
    ):
        mut = MinimizeTiltMutation(closest_distances=closest_distances, max_retries=1)
        mutation_reproducability_assessment(ind_crystal, mut, assess_change=False)


class TestRotationMutation:
    def test_mutate(
        self,
        closest_distances,
        ind_crystal,
        mutation_reproducability_assessment,
    ):
        mut = RotationMutation(closest_distances=closest_distances, max_retries=1)
        mutation_reproducability_assessment(ind_crystal, mut, assess_change=True)
