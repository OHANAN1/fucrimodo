# TODO: Fix weird error
#       ^- Wow, nobody wants to read such a thing...
#           I forgot the bug, so here is a seahorse:
#
#       \/)/)
#     _'  oo(_.-.
#   /'.     .---'
# /'-./    (
# )     ; __\
# \_.'\ : __|
#      )  _/
#     (  (,.
#   mrf'-.-'
#
import matid
import numpy as np

from ....core import Individual
from ....core.utils import CustomClosestDistances

from .abstract import Mutation


class GetConventionalCellMutation(Mutation):
    """
    Convert an individual to its conventional cell using symmetry analysis.

    This mutation is based on the ``matid.SymmetryAnalyzer`` class. It
    performs a symmetry analysis on the individual and constructs a new
    individual with the conventional cell, atomic positions, and atomic
    numbers.

    The mutation is deterministic, so retrying it will produce the same
    result.

    For more info refer to the `MatID Documentation
    <https://singroup.github.io/matid/index.html>`__.

    :param closest_distances: Minimum allowed interatomic distances used
        to validate the mutated structure.
    :type closest_distances: CustomClosestDistances
    :param symmetry_tol: Tolerance used in the symmetry analysis. Defaults
        to ``1e-5``.
    :type symmetry_tol: float | None
    :param max_volume_increase: Maximum allowed volume increase relative
        to the original cell. Defaults to ``1.2``.
    :type max_volume_increase: float
    :param max_volume_decrease: Maximum allowed volume decrease relative
        to the original cell. Defaults to ``0.8``.
    :type max_volume_decrease: float
    :param max_retries: Maximum number of attempts to produce a valid
        mutation. Not recommended to increase above ``1`` since the
        mutation is deterministic. Defaults to ``100``.
    :type max_retries: int
    :param rng: Random number generator. If ``None``, the base class
        creates one.
    :type rng: None | np.random.Generator
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        symmetry_tol: float | None = 1e-5,
        max_volume_increase: float = 1.2,
        max_volume_decrease: float = 0.8,
        max_retries: int = 100,
        rng: None | np.random.Generator = None,
    ):
        super().__init__(
            closest_distances=closest_distances, max_retries=max_retries, rng=rng
        )

        self.symmetry_tol = symmetry_tol
        self.max_volume_increase = max_volume_increase
        self.max_volume_decrease = max_volume_decrease

    def _perform_mutation(self, individual: Individual) -> Individual | None:
        # Get the original volume
        original_volume = individual.get_volume()

        individual.set_pbc([True, True, True])

        # Analyze the individual to get the conventional cell
        analyzer = matid.SymmetryAnalyzer(individual.copy(), self.symmetry_tol)
        conventional_cell = analyzer.get_conventional_system()

        # Create a new individual with the analyzed properties
        positions = conventional_cell.get_positions()
        atomic_numbers = conventional_cell.get_atomic_numbers()
        cell = conventional_cell.get_cell()
        offspring = Individual(atomic_numbers, positions=positions, cell=cell)

        # Check if the volume increase is too large
        if offspring.get_volume() > original_volume * self.max_volume_increase:
            return None

        # Check if the volume decrease is too large
        if offspring.get_volume() < original_volume * self.max_volume_decrease:
            return None

        # Check if the new individual is the same as the original
        # If so, return None to symbolize that the mutation was not successful
        if offspring == individual:
            return None

        return offspring
