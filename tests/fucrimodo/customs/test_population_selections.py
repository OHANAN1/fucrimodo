import numpy as np
import pytest

from fucrimodo.core.utils.fitness_utils import assign_fitness_to_individuals
from fucrimodo.customs.population_selections import (
    NSGA2Selection,
    RandomSelection,
    TournamentDCDSelection,
    TournamentSelection,
)


class TestRandomSelection:
    def test___repr__(self):
        assert RandomSelection().__repr__() == "RandomSelection()"

    def test_select(
        self,
        ind_slab,
        ind_crystal,
        ind_molecule,
        example_fitness,
    ):
        individuals = [ind_slab, ind_crystal, ind_molecule]
        assign_fitness_to_individuals(
            individuals=individuals, fitness_functions=example_fitness
        )

        sel_1 = RandomSelection(rng=np.random.default_rng(42)).select(individuals, n=2)
        assert len(sel_1) == 2

        # Assert that selection is reproducable
        sel_2 = RandomSelection(rng=np.random.default_rng(42)).select(individuals, n=2)
        assert sel_1 == sel_2

        # Assert that selection with different seed can change
        sel_3 = RandomSelection(rng=np.random.default_rng(43)).select(individuals, n=1)
        assert sel_2 != sel_3


class TestTournamentSelection:
    @pytest.fixture()
    def tournament_selection(self):
        return TournamentSelection(tournament_size=2, rng=np.random.default_rng(42))

    def test___repr__(self, tournament_selection: TournamentSelection):
        assert (
            tournament_selection.__repr__() == "TournamentSelection(tournament_size=2)"
        )

    def test_select(
        self,
        ind_slab,
        ind_crystal,
        ind_molecule,
        example_fitness,
    ):
        individuals = [ind_slab, ind_crystal, ind_molecule]
        assign_fitness_to_individuals(
            individuals=individuals, fitness_functions=example_fitness
        )

        sel_1 = TournamentSelection(
            tournament_size=2, rng=np.random.default_rng(42)
        ).select(individuals, n=1)
        assert len(sel_1) == 1

        # Assert that selection is reproducable
        sel_2 = TournamentSelection(
            tournament_size=2, rng=np.random.default_rng(42)
        ).select(individuals, n=1)
        assert sel_1 == sel_2

        # Assert that selection with different seed can change
        sel_3 = TournamentSelection(
            tournament_size=2, rng=np.random.default_rng(43)
        ).select(individuals, n=1)
        assert sel_2 != sel_3


class TestNSGA2Selection:
    @pytest.fixture()
    def nsga2_selection(self):
        return NSGA2Selection(nondominated_sorting="standard")

    def test___repr__(self, nsga2_selection: NSGA2Selection):
        assert (
            nsga2_selection.__repr__()
            == "NSGA2Selection(nondominated_sorting=standard)"
        )

    def test_select(
        self,
        nsga2_selection: NSGA2Selection,
        ind_slab,
        ind_crystal,
        ind_molecule,
        example_fitness,
    ):
        individuals = [ind_slab, ind_crystal, ind_molecule]
        assign_fitness_to_individuals(
            individuals=individuals, fitness_functions=example_fitness
        )

        sel_1 = nsga2_selection.select(individuals, 2)
        assert len(sel_1) == 2
        assert sel_1 == [ind_crystal, ind_slab]

        # Check if its deterministic
        sel_2 = nsga2_selection.select(individuals, 2)
        assert sel_1 == sel_2


class TestTournamentDCDSelection:
    @pytest.fixture()
    def tournament_dcd_selection(self):
        return TournamentDCDSelection(rng=np.random.default_rng(42))

    def test___repr__(self, tournament_dcd_selection):
        assert tournament_dcd_selection.__repr__() == "TournamentDCDSelection()"

    def test_select(
        self,
        ind_slab,
        ind_crystal,
        ind_molecule,
        example_fitness,
    ):
        individuals = [
            ind_slab,
            ind_crystal,
            ind_molecule,
            ind_molecule,
            ind_slab,
            ind_crystal,
            ind_molecule,
            ind_molecule,
        ]
        assign_fitness_to_individuals(
            individuals=individuals, fitness_functions=example_fitness
        )

        sel_1 = TournamentDCDSelection(rng=np.random.default_rng(42)).select(
            individuals, n=4
        )
        assert len(sel_1) == 4

        # Assert that selection is reproducable
        sel_2 = TournamentDCDSelection(rng=np.random.default_rng(42)).select(
            individuals, n=4
        )
        assert sel_1 == sel_2

        # Assert that selection with different seed can change
        sel_3 = TournamentDCDSelection(rng=np.random.default_rng(43)).select(
            individuals, n=4
        )
        assert sel_2 != sel_3

        # Assert that n must be minimum 4 else error is raised
        with pytest.raises(ValueError):
            TournamentDCDSelection().select(individuals, n=3)

        # Assert that minimum 4 individuals must be there
        with pytest.raises(AssertionError):
            TournamentDCDSelection().select(individuals[:3], n=3)
