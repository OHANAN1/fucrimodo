import pytest

import numpy as np
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds


@pytest.fixture
def example_cellbounds():
    return CustomCellBounds(
        {
            "phi": [20, 160],
            "chi": [60, 120],
            "psi": [20, 160],
            "a": [2, 6],
            "b": [2, 6],
            "c": [2, 6],
        }
    )


def test_repr(example_cellbounds):
    assert hasattr(example_cellbounds, "__repr__")


def test_is_within_bounds(example_cellbounds, ind_crystal, ind_slab):
    assert example_cellbounds.is_within_bounds(ind_crystal.cell)
    assert not example_cellbounds.is_within_bounds(ind_slab.cell)


def test_ind_is_within_bounds(example_cellbounds, ind_crystal, ind_slab, ind_molecule):
    assert example_cellbounds.ind_is_within_bounds(ind_crystal)
    assert not example_cellbounds.ind_is_within_bounds(ind_slab)

    # Raises assertion error if individual does not have a cell
    with pytest.raises(AssertionError):
        example_cellbounds.ind_is_within_bounds(ind_molecule)


def test_bounds_property(example_cellbounds):
    assert hasattr(example_cellbounds, "bounds")
    assert example_cellbounds.bounds["a"] == [2, 6]

    # Degrees will be converted to rad
    assert example_cellbounds.bounds["chi"] == [60 / 180 * np.pi, 120 / 180 * np.pi]
