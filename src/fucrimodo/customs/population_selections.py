from typing import Any, Callable, Literal

import numpy as np
from deap import tools
from operator import attrgetter
from ..core import Individual
from ..core.abstracts import (
    PopulationSelection,
)


class RandomSelection(PopulationSelection):
    """Select individuals uniformly at random from a population.

    Selection is performed with replacement, so the same individual may
    appear multiple times in the returned list.

    :param rng: Random number generator. If ``None``, a new default
        generator is created.
    """

    def __init__(
        self,
        rng: np.random.Generator | None = None,
    ):
        if not rng:
            rng = np.random.default_rng()
        self._rng = rng

    def select(self, individuals: list[Individual], n: int) -> list[Individual]:
        idx = self._rng.integers(len(individuals), size=n)
        return [individuals[i] for i in idx]

    def __repr__(self):
        return "RandomSelection()"


class TournamentSelection(PopulationSelection):
    """Select individuals using tournament selection.

    Each selected individual is determined by sampling
    ``tournament_size`` aspirants uniformly at random from the population
    (with replacement) and choosing the one with the highest fitness.

    :param tournament_size: Number of individuals in each tournament.
    :param rng: Random number generator. If ``None``, a new default
        generator is created.
    """

    def __init__(
        self,
        tournament_size: int,
        rng: np.random.Generator | None = None,
    ):
        if not rng:
            rng = np.random.default_rng()
        self._rng = rng
        self._random_selection = RandomSelection(rng=rng)

        self._tournament_size = tournament_size

    def __repr__(self) -> str:
        return f"TournamentSelection(tournament_size={self._tournament_size})"

    def select(self, individuals: list[Individual], n: int) -> list[Individual]:
        chosen = []
        for _ in range(n):
            aspirants = self._random_selection.select(
                individuals, self._tournament_size
            )
            chosen.append(max(aspirants, key=attrgetter("fitness")))
        return chosen


class NSGA2Selection(PopulationSelection):
    """Select individuals using the NSGA-II multi-objective selection algorithm.

    This is a wrapper around :func:`deap.tools.selNSGA2`. The selector
    performs non-dominated sorting on the input individuals and uses
    crowding distance to pick ``n`` diverse, Pareto-optimal individuals.

    The :attr:`nondominated_sorting` attribute controls the non-dominated
    sorting algorithm:

    - ``'standard'`` (default): Deb's Fast Non-Dominated Sorting approach,
      with time complexity O(M N^2) where M is the number of
      objectives and N is the number of individuals.
    - ``'log'``: Fortin et al.'s Generalized Reduced Run-Time Complexity
      Non-Dominated Sorting algorithm, which can be faster for large
      populations.

    See the `DEAP docs
    <https://deap.readthedocs.io/en/master/api/tools.html>`__ for more details.

    :param nondominated_sorting: Non-dominated sorting algorithm to use.
        Must be either ``'standard'`` or ``'log'``. Defaults to ``'standard'``.
    :type nondominated_sorting: str

    :raises ValueError: If ``nondominated_sorting`` is not ``'standard'``
        or ``'log'``.
    """

    def __init__(
        self,
        nondominated_sorting: Literal["log", "standard"] = "standard",
    ):
        if not nondominated_sorting in ["log", "standard"]:
            raise ValueError(
                "Only 'standard' and 'log' is allowed for param 'nondominated_sorting'!"
            )
        self._nondominated_sorting = nondominated_sorting

    def __repr__(self) -> str:
        return f"NSGA2Selection(nondominated_sorting={self._nondominated_sorting})"

    def select(self, individuals: list[Individual], n: int) -> list[Individual]:
        return tools.selNSGA2(individuals, k=n, nd=self._nondominated_sorting)


class TournamentDCDSelection(PopulationSelection):
    """Select individuals using tournament selection based on dominance and crowding distance.

    This is a reimplementation of DEAP's ``selTournamentDCD`` that uses an
    explicit :class:`numpy.random.Generator` instead of Python's global random
    module. The algorithm selects individuals by repeatedly running pairwise
    tournaments between groups from two independent shuffles of the population.
    In each tournament, the winner is determined by:

    1. Pareto dominance: the individual whose fitness dominates the other wins.
    2. Crowding distance: if neither dominates, the individual with the larger
       crowding distance wins.
    3. Random tie-break: if both have the same crowding distance, one is chosen
       at random.

    The population automatically gets the crowding distances assigned via DEAP's
    :func:`deap.tools.emo.assignCrowdingDist`. After it the parameter is deleted.

    See the `DEAP docs
    <https://deap.readthedocs.io/en/master/api/tools.html>`__ for more details.

    :param sort_by: Key function used to sort the selected individuals after
        selection. If ``None``, the selected individuals are returned in the
        order they were chosen. Defaults to sorting by fitness values.
    :param rng: Random number generator. If ``None``, a new default generator
        is created.

    :raises ValueError: If ``n < 4`` or ``n > len(individuals)``.
    :raises ValueError: If ``n == len(individuals)`` and ``n`` is not divisible
        by four.
    :raises AssertionError: If the population has fewer than 4 individuals.
    """

    def __init__(
        self,
        sort_by: Callable[[Individual], Any] | None = lambda x: x.fitness.values,
        rng: np.random.Generator | None = None,
    ):
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng

        self.sort_by = sort_by

    def _np_selTournamentDCD(self, individuals: list[Individual], n: int):
        """Tournament selection based on dominance (D) and crowding distance (CD).

        Same semantics as DEAP's selTournamentDCD, but uses an explicit
        np.random.Generator instead of Python's global random module.
        """
        if n > len(individuals):
            raise ValueError(
                "selTournamentDCD: n must be less than or equal to individuals length"
            )

        if n < 4:
            raise ValueError("n must be bigger or equal to 4")

        if n == len(individuals) and n % 4 != 0:
            raise ValueError(
                "selTournamentDCD: n must be divisible by four if k == len(individuals)"
            )

        def tourn(ind1, ind2):
            if ind1.fitness.dominates(ind2.fitness):
                return ind1
            elif ind2.fitness.dominates(ind1.fitness):
                return ind2

            if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
                return ind2
            elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
                return ind1

            if self._rng.random() <= 0.5:
                return ind1
            return ind2

        # Two independent shuffles of the population indices
        idx1 = self._rng.permutation(len(individuals))
        idx2 = self._rng.permutation(len(individuals))

        chosen = []
        for i in range(0, n, 4):
            chosen.append(tourn(individuals[idx1[i]], individuals[idx1[i + 1]]))
            chosen.append(tourn(individuals[idx1[i + 2]], individuals[idx1[i + 3]]))
            chosen.append(tourn(individuals[idx2[i]], individuals[idx2[i + 1]]))
            chosen.append(tourn(individuals[idx2[i + 2]], individuals[idx2[i + 3]]))

        return chosen

    def select(self, individuals: list[Individual], n: int) -> list[Individual]:
        assert len(individuals) >= 4

        # Assign crowding distance to each individual
        tools.emo.assignCrowdingDist(individuals)

        # Select individuals using the selTournamentDCD function
        chosen = self._np_selTournamentDCD(individuals=individuals, n=n)

        # Remove the crowding distance of the selected individuals
        for individual in chosen:
            if hasattr(individual.fitness, "crowding_dist"):
                del individual.fitness.crowding_dist

        if self.sort_by is not None:
            chosen.sort(
                key=self.sort_by,
            )

        return chosen

    def __repr__(self) -> str:
        return "TournamentDCDSelection()"
