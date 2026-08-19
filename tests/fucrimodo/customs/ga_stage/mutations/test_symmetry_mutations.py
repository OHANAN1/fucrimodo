import pytest

from fucrimodo.customs.ga_stage.mutations.symmetry_mutations import (
    GetConventionalCellMutation,
)


class TestGetConvenionalCellMutation:
    def test_mutate(
        self,
        closest_distances,
        ind_crystal,
        mutation_reproducability_assessment,
    ):

        cell = ind_crystal.cell
        cell[0] += 0.1
        cell[1] += 0.1
        cell[2] += 0.1

        ind_crystal.set_cell(cell)

        mut = GetConventionalCellMutation(
            closest_distances=closest_distances,
            symmetry_tol=0.3,
            max_volume_increase=4.0,
            max_volume_decrease=0.8,
            max_retries=100,
        )

        mutation_reproducability_assessment(ind_crystal, mut, assess_change=False)
