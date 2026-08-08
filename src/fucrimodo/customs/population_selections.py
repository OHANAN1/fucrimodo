from typing import Callable

import numpy as np
from deap import tools
from operator import attrgetter
from ..core import Individual
from ..core.abstracts import (
    PopulationSelection,
)


class RandomSelection(PopulationSelection):
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


class TournamentSelection(PopulationSelection):
    """Selects a population using the tournament selection algorithm.

    Wrapper around the DEAP tournament selection algorithm. More information
    about the algorithm can be found in the DEAP documentation:
    `https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.selTournament`

    :param k: The number of individuals to select. Either integer for total
        number or float for percentage of the population.
    :param tournsize: The number of individuals participating in each tournament.
    """

    def __init__(
        self,
        tournament_size: int,
        rng: np.random.Generator | None = None,
    ):
        self._tournament_size = tournament_size
        self.random_selection = RandomSelection(rng=rng)

    def __repr__(self) -> str:
        return f"TournamentSelection(tournament_size={self._tournament_size})"

    def select(self, individuals: list[Individual], n: int) -> list[Individual]:
        """Selects a population using the tournament selection.

        :param individuals: A list of individuals to select from. The
            individual must have the :class:`FitnessStorage` class at
            :attr:`Individual.fitness` to be used in the tounament selection.
        :param n: The number of individuals to select.

        :return: A list of individuals selected using the tounament selection.
        """
        chosen = []
        for _ in range(n):
            aspirants = self.random_selection.select(individuals, self._tournament_size)
            chosen.append(max(aspirants, key=attrgetter("fitness")))
        return chosen


class NSGA2Selection(PopulationSelection):
    """Selects a population using the NSGA-II algorithm.

    Wrapper around the DEAP NSGA-II algorithm. More information about the
    algorithm can be found in the DEAP documentation:
    `https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.selNSGA2`

    :param nondominated_sorting: The method used for non-dominated sorting.
        Options are 'standard' and 'log'. See DEAP documentation for more
        information.
    """

    def __init__(
        self,
        nondominated_sorting: str = "standard",
    ):
        self._nondominated_sorting = nondominated_sorting

    def __repr__(self) -> str:
        return f"NSGA2Selection(nondominated_sorting={self._nondominated_sorting})"

    def select(self, individuals: list[Individual], n: int) -> list[Individual]:
        """Selects a population using the NSGA-II algorithm.

        :param individuals: A list of individuals to select from. The
            individual must have the :class:`FitnessStorage` class at
            :attr:`Individual.fitness` to be used in the NSGA-II algorithm.
        :param n: The number of individuals to select.

        :return: A list of individuals selected using the NSGA-II algorithm.
            Sorted by the NSGA-II algorithm from best to worst.
        """
        # Perform NSGA-II selection.
        individuals = tools.selNSGA2(individuals, k=n, nd=self._nondominated_sorting)

        return individuals


class TournamentDCDSelection(PopulationSelection):
    """Selects a population using the tournament selection algorithm with DCD.

    Wrapper around the DEAP tournament selection algorithm with DCD. More
    information about the algorithm can be found in the DEAP documentation:
    `https://deap.readthedocs.io/en/master/api/tools.html#deap.tools.selTournamentDCD`

    :param sort_by: A function that sorts the selected individuals. The
        function should take an individual as input and return a value to
        sort by. If None, the selected individuals are not sorted.
    """

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        sort_by: Callable[[Individual], float] | None = lambda x: x.fitness.values,
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
                "selTournamentDCD: k must be less than or equal to individuals length"
            )

        if n == len(individuals) and n % 4 != 0:
            raise ValueError(
                "selTournamentDCD: k must be divisible by four if k == len(individuals)"
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
        # Assign crowding distance to each individual, as expected by the
        # selTournamentDCD function
        tools.emo.assignCrowdingDist(individuals)

        # Select individuals using the selTournamentDCD function
        chosen = self._np_selTournamentDCD(individuals=individuals, n=n)

        # Reset the crowding distance of the selected individuals
        for individual in chosen:
            if hasattr(individual, "crowding_dist"):
                del individual.crowding_dist

        if self.sort_by is not None:
            chosen.sort(
                key=self.sort_by,
            )

        return chosen

    def __repr__(self) -> str:
        return "TournamentDCDSelection()"
