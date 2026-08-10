import pytest
from fucrimodo.core.utils.closest_distances_class import (
    CustomClosestDistances,
)
from fucrimodo.core import Individual


@pytest.fixture
def example_closest_distances():
    return CustomClosestDistances(species=["H", "He"], ratio_of_covalent_radii=1.0)


def test_initialization(example_closest_distances):
    # Check if atomic numbers and chemical elements are assigned properly
    assert example_closest_distances._chemical_symbols == ["H", "He"]
    assert example_closest_distances._atomic_numbers == [1, 2]

    # Initialize with atomic numbers
    ccd = CustomClosestDistances(species=[1, 2], ratio_of_covalent_radii=1.0)

    # Test that this produces same object
    assert example_closest_distances == ccd

    # Check if atomic numbers and chemical elements are also assigned properly
    assert ccd._chemical_symbols == ["H", "He"]
    assert ccd._atomic_numbers == [1, 2]


def test_class_works_like_is_dict(example_closest_distances):
    assert example_closest_distances.get((1, 1))
    assert example_closest_distances.get((1, 2))
    assert len(example_closest_distances) == 4  # all len-2 combinations of {1,2}


def test_atoms_are_too_close(example_closest_distances: CustomClosestDistances):
    ind = Individual(
        ["H", "H"],
        positions=[[0, 0, 0], [0, 0, 0.5]],
    )
    assert example_closest_distances.atoms_are_too_close(ind)

    # Test if class properly accounts for atoms in neighboring cells
    ind = Individual(
        ["H", "H"],
        positions=[[0, 0, 0], [0, 0, 1]],
        cell=[1, 0.2, 1],
        pbc=True,
    )
    assert example_closest_distances.atoms_are_too_close(ind)


def test_repr(example_closest_distances):
    assert (
        example_closest_distances.__repr__()
        == "CustomClosestDistances(chemical_symbols=['H', 'He'], ratio_of_covalent_radii=1.0)"
    )
