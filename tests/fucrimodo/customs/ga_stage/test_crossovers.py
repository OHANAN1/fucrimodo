import numpy as np
import pytest

from fucrimodo.core import Individual
from fucrimodo.customs.ga_stage.crossovers import (
    Crossover,
    CutAndSpliceCrossover,
    OnePointElementCrossover,
    OnePointPositionCrossover,
    StackCellsCrossover,
    UnitCellCrossover,
)


def reproducability_assesment(
    par1: Individual,
    par2: Individual,
    cross: Crossover,
    check_only_1=True,
):

    original_state = cross._rng.bit_generator.state

    # run multiple tests
    c1_structures = []
    c2_structures = []
    required_steps_list = []
    successes = []
    for _ in range(5):
        c1, c2, success = cross.crossover(par1.copy(), par2.copy())
        c1_structures.append(c1)
        c2_structures.append(c2)
        successes.append(success)
        if success:
            assert c1 != par1 or c2 != par2

        required_steps_list.append(cross.required_steps)

    # Check if there was at least one success
    assert any(successes)

    # Check that some mutations took more than one step
    if check_only_1:
        assert any(s > 1 for s in required_steps_list)

    # Check reproducability
    # Restore original big generator state
    cross._rng.bit_generator.state = original_state
    for i in range(5):
        c1, c2, success = cross.crossover(par1.copy(), par2.copy())
        assert success == successes[i]
        assert c1 == c1_structures[i]
        assert c2 == c2_structures[i]

    # Check that results change
    c1_changed = []
    for i in range(5):
        c1, _, _ = cross.crossover(par1.copy(), par2.copy())
        c1_changed.append(c1 != c1_structures[i])
    assert any(c1_changed)


class MockCrossover(Crossover):
    """
    This Crossover just returns copies of the parents.
    Use crossover_probability in genetic algorithm rather than this
    class to turn off crossover.
    """

    def _perform_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> tuple[Individual, Individual] | tuple[None, None]:
        return (parent1.copy(), parent2.copy())


class TestCrossover:
    def test_incomplete_child(self, closest_distances):
        class Incomplete(Crossover):
            pass

        with pytest.raises(TypeError):
            Incomplete(closest_distances=closest_distances)  # type: ignore

    def test_minimal_class(self, closest_distances, ind_slab, ind_molecule):
        mock_cross = MockCrossover(closest_distances)

        ind_slab_cp, ind_molecule_cp, success = mock_cross.crossover(
            ind_slab, ind_molecule
        )
        assert success
        assert ind_slab == ind_slab_cp
        assert ind_molecule == ind_molecule_cp


class TestUnitCellCrossover:
    def test_crossover(
        self,
        closest_distances,
        ind_crystal,
    ):
        unit_cell_cross = UnitCellCrossover(
            closest_distances=closest_distances,
            scale_atoms=True,
            max_retries=10,
            rng=np.random.default_rng(42),
        )

        # slightly change unitcell
        changed_ind_crystal = ind_crystal.copy()
        cell = changed_ind_crystal.cell[:]
        cell[0] += 0.5
        changed_ind_crystal.set_cell(cell)

        c1, c2, success = unit_cell_cross.crossover(
            ind_crystal.copy(), changed_ind_crystal.copy()
        )
        assert success
        assert c1 != ind_crystal or c2 != changed_ind_crystal

        # Test that atoms get scaled
        assert np.any(c1.positions != ind_crystal.positions)

        # Check reproducability
        unit_cell_cross = UnitCellCrossover(
            closest_distances=closest_distances,
            scale_atoms=True,
            max_retries=10,
            rng=np.random.default_rng(42),
        )

        c1_again, c2_again, success = unit_cell_cross.crossover(
            ind_crystal.copy(), changed_ind_crystal.copy()
        )
        assert success
        assert c1 == c1_again
        assert c2 == c2_again

        # Check that results change
        c1_diff, c2_diff, success = unit_cell_cross.crossover(
            ind_crystal.copy(), changed_ind_crystal.copy()
        )
        assert c1_diff != c1 or c2_diff != c2


class TestStackCellsCrossover:
    def test_crossover(
        self,
        closest_distances,
        ind_crystal,
        cell_bounds,
    ):
        cross = StackCellsCrossover(
            closest_distances=closest_distances,
            scale_atoms=True,
            max_retries=10,
            cell_bounds=cell_bounds,
            rng=np.random.default_rng(42),
        )

        # slightly change unitcell
        changed_ind_crystal = ind_crystal.copy()
        cell = changed_ind_crystal.cell[:]
        cell[0] += 0.5
        changed_ind_crystal.set_cell(cell)

        c1, c2, success = cross.crossover(
            ind_crystal.copy(), changed_ind_crystal.copy()
        )
        assert success
        assert c1 != ind_crystal or c2 != changed_ind_crystal
        assert len(c1) == len(ind_crystal) * 2
        assert len(c2) == len(changed_ind_crystal) * 2

        # Check reproducability
        cross = StackCellsCrossover(
            closest_distances=closest_distances,
            scale_atoms=True,
            max_retries=10,
            cell_bounds=cell_bounds,
            rng=np.random.default_rng(42),
        )

        c1_again, c2_again, success = cross.crossover(
            ind_crystal.copy(), changed_ind_crystal.copy()
        )
        assert success
        assert c1 == c1_again
        assert c2 == c2_again

        # Check that results change
        c1_inds = []
        for _ in range(5):
            c1_diff, _, _ = cross.crossover(
                ind_crystal.copy(), changed_ind_crystal.copy()
            )
            c1_inds.append(c1_diff)
        assert any([ind != c1 for ind in c1_inds])


class TestOnePointElementCrossover:
    def test_crossover(
        self,
        closest_distances,
        ind_molecule,
    ):
        cross = OnePointElementCrossover(
            closest_distances=closest_distances,
            max_retries=10,
            rng=np.random.default_rng(42),
        )

        # slightly change unitcell
        changed_ind = ind_molecule.copy()
        atomic_num = changed_ind.numbers
        atomic_num[0] = 2
        atomic_num[1] = 3
        changed_ind.set_atomic_numbers(atomic_num)

        c1, c2, success = cross.crossover(ind_molecule.copy(), changed_ind.copy())
        assert success
        assert c1 != ind_molecule or c2 != changed_ind
        assert len(c1) == len(ind_molecule)
        assert len(c2) == len(changed_ind)

        # Check reproducability
        cross = OnePointElementCrossover(
            closest_distances=closest_distances,
            max_retries=10,
            rng=np.random.default_rng(42),
        )

        c1_again, c2_again, success = cross.crossover(
            ind_molecule.copy(), changed_ind.copy()
        )
        assert success
        assert c1 == c1_again
        assert c2 == c2_again

        # Check that results change
        c1_inds = []
        for _ in range(10):
            c1_diff, _, _ = cross.crossover(ind_molecule.copy(), changed_ind.copy())
            c1_inds.append(c1_diff)
        assert any([ind != c1 for ind in c1_inds])

        # Test with different length structures
        two_atomic_ind = Individual(["He", "He"], [[0, 0, 0], [2, 0, 0]])
        successes = []
        for _ in range(10):
            c1, c2, success = cross.crossover(
                two_atomic_ind.copy(), ind_molecule.copy()
            )
            successes.append(success)
        assert any(successes)

        # Test that with one atomic structure not working
        two_atomic_ind = Individual(["He"], [[0, 0, 0]])
        successes = []
        for _ in range(5):
            c1, c2, success = cross.crossover(
                two_atomic_ind.copy(), ind_molecule.copy()
            )
            successes.append(success)
        assert not any(successes)

        rng = np.random.default_rng(42)

        # Test normal crossover
        cross = OnePointElementCrossover(
            closest_distances=closest_distances,
            max_retries=10,
            rng=np.random.default_rng(42),
        )

        new_ind = ind_molecule.copy()
        numbers = new_ind.numbers
        numbers[0] = 8
        numbers[1] = 8
        new_ind.set_atomic_numbers(numbers)

        # TODO: Test it later, keep it open now, so I remember
        reproducability_assesment(
            par1=ind_molecule, par2=new_ind, cross=cross, check_only_1=False
        )


class TestOnePointPositionCrossover:

    def test_crossover(
        self,
        closest_distances,
        ind_molecule,
    ):
        rng = np.random.default_rng(42)

        # Test normal crossover
        cross = OnePointPositionCrossover(
            closest_distances=closest_distances,
            max_retries=10,
            rng=rng,
        )

        # slightly change unitcell
        changed_ind = ind_molecule.copy()
        pos = changed_ind.positions
        pos[0] -= 0.5
        pos[1] -= 0.4
        pos[2] -= 0.3
        changed_ind.set_positions(pos)

        reproducability_assesment(
            par1=ind_molecule, par2=changed_ind, cross=cross, check_only_1=False
        )

        # Test with different length structures
        two_atomic_ind = Individual(["He", "He"], [[0, 0, 0], [2, 0, 0]])
        successes = []
        for _ in range(5):
            c1, c2, success = cross.crossover(
                two_atomic_ind.copy(), ind_molecule.copy()
            )
            successes.append(success)
        assert any(successes)

        # Test that with one atomic structure not working
        two_atomic_ind = Individual(["He"], [[0, 0, 0]])
        successes = []
        for _ in range(5):
            c1, c2, success = cross.crossover(
                two_atomic_ind.copy(), ind_molecule.copy()
            )
            successes.append(success)
        assert not any(successes)


class TestCutAndSpliceCrossover:
    def test_crossover(
        self,
        closest_distances,
        cell_bounds,
        ind_crystal,
    ):
        rng = np.random.default_rng(42)

        # Test normal crossover
        cross = CutAndSpliceCrossover(
            cell_bounds=cell_bounds,
            closest_distances=closest_distances,
            max_retries=1,
            n_top="all",
            number_of_variable_cell_vectors=3,
            rng=rng,
        )

        new_ind = ind_crystal.copy()
        cell = ind_crystal.cell
        cell[0] += 0.5
        new_ind.set_cell(cell)
        pos = ind_crystal.positions
        pos[0] += 0.5
        new_ind.set_positions(pos)

        reproducability_assesment(
            par1=ind_crystal, par2=new_ind, cross=cross, check_only_1=False
        )
