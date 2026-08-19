import numpy as np
from ase.cell import Cell
from ase_ga.utilities import CellBounds

from ..individual import Individual


class CustomCellBounds:
    """Define and check the max/min allowed size of a unitcell.

    This class works like :class:`ase_ga.utilities.CellBounds` from the `ASE library <https://github.com/dtu-energy/ase-ga>`__, but with custom
    functionality:

    * has __repr__ method to print the bounds
    * bounds is a property
    * has method `ind_is_within_bounds` to test if individual object is within bounds

    Example use:

    .. code-block:: python

        CustomCellBounds(
            bounds={
                'phi': [20, 160],
                'chi': [60, 120],
                'psi': [20, 160],
                'a': [2, 20],
                'b': [2, 20],
                'c': [2, 20]
            }
        )

    """

    def __init__(self, bounds: dict[str, list[float]] = {}):
        self._ase_cellbounds = CellBounds(bounds)
        self._bounds = self._ase_cellbounds.bounds

    def is_within_bounds(self, cell: Cell):
        return self._ase_cellbounds.is_within_bounds(cell)

    def ind_is_within_bounds(self, individual: Individual):
        assert hasattr(
            individual, "cell"
        ), "Only individuals with cell can be within the cell bound."
        assert not np.all(
            individual.cell[:] == 0.0  # type: ignore
        ), "Cell of individual is not valid. Is it a molecule?"
        return self.is_within_bounds(individual.cell)

    @property
    def bounds(self) -> dict[str, list[float]]:
        return self._bounds

    def __repr__(self):
        return f"CustomCellBounds(bounds={self._bounds})"
