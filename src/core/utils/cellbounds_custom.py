from ase.ga.utilities import CellBounds


class CustomCellBounds():
    """
    Basically works like ase.ga.utilities.CellBounds, but with custom
    functionality:
    - has better __repr__ method to print the bounds
    - bounds is a property

    Example use:
    >>> CustomCellBounds(
    ...     bounds={
    ...         'phi': [20, 160],
    ...         'chi': [60, 120],
    ...         'psi': [20, 160],
    ...         'a': [2, 20],
    ...         'b': [2, 20],
    ...         'c': [2, 20]
    ...     }
    ... )
    """

    def __init__(self, bounds: dict[str, list[float]] = {}):
        self._ase_cellbounds = CellBounds(bounds)
        self._bounds = self._ase_cellbounds.bounds

    def is_within_bounds(self, cell):
        return self._ase_cellbounds.is_within_bounds(cell)

    @property
    def bounds(self) -> dict[str, list[float]]:
        return self._bounds

    def __repr__(self):
        return f"CustomCellBounds(bounds={self._bounds})"
