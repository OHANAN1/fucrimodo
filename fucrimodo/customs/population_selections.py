import random
from typing import Callable, Optional, Sequence
from fucrimodo.core.modules.individual import Individual
from fucrimodo.core.modules.population_generator import PopulationGenerator
from fucrimodo.customs.population_generator import OneAtomicCrystalGenerator
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import rbf_kernel
import warnings
from fucrimodo.core.modules import PopulationSelection
from deap import tools

# ╒══════════════════════════════════════════════════════════╕
#                    StartPopulation Class
# ╘══════════════════════════════════════════════════════════╛

class RandomSelectionPopulation(PopulationSelection):
    """
    A class that returns a list of n random crystals.
    Set database_name to describe the database the original crystals are from.

    Call the object after initialization to get the list of crystals.
    """

    def __init__(
        self,
        n: int,
        database_name: Optional[str] = None
    ):
        self.n = n
        self.database_name = database_name

    def add_individuals(self, individuals: list[Individual]) -> None:
        """
        Adds a list of crystals to the start population.
        """
        self.individuals = individuals

    def get_individual(self) -> Individual:
        """
        Returns n random crystals from the crystals list.
        """
        chosen_individuals = random.choice(self.individuals)
        return chosen_individuals

    def select_start_pop(
        self,
        individuals: list[Individual],
    ) -> list[Individual]:
        """
        Returns n random crystals from the crystals list.
        """
        chosen_individuals = random.choices(individuals, k=self.n)
        return chosen_individuals

    def __repr__(self) -> str:
        if self.database_name is not None:
            return "RandomSelectionPopulation(n={}, database_name={})".format(
                self.n, self.database_name
            )
        else:
            return "RandomSelectionPopulation(n={})".format(self.n)


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
    ):
        self._tournament_size = tournament_size

    def __repr__(self) -> str:
        return f"TournamentSelection(tournsize={self._tournament_size})"

    def select(self, individuals: list[Individual], n: int) -> list[Individual]:
        """Selects a population using the tournament selection.

        :param individuals: A list of individuals to select from. The 
            individual must have the :class:`FitnessStorage` class at 
            :attr:`Individual.fitness` to be used in the tounament selection.
        :param n: The number of individuals to select.

        :return: A list of individuals selected using the tounament selection.
        """
        # Perform tournament selection.
        individuals = tools.selTournament(
            individuals, k=n, tournsize=self._tournament_size
        )

        return individuals


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
        individuals = tools.selNSGA2(
            individuals, k=n, nd=self._nondominated_sorting
        )

        return individuals


class BestCrystalsPopulation(PopulationSelection):
    """
    A class that returns a list of n crystals with the highest fitness.
    Set database_name to describe the database the original crystals are from.

    Call the object after initialization to get the list of crystals.
    """

    def __init__(
        self,
        n: int,
        evaluation_function: Callable[[Individual], float],
        database_name: Optional[str] = None,
        verbose: int = 1
    ):
        self.n = n
        self.evaluation_function = evaluation_function
        self.database_name = database_name
        self.verbose = verbose

    def __get_n_best_individuals(
        self,
        individuals: list[Individual],
        verbose: int = 1
    ) -> list[Individual]:
        fitness_list = [self.evaluation_function(
            individual) for individual in individuals]
        sorted_list = sorted(
            zip(individuals, fitness_list),
            key=lambda x: x[1],
            reverse=True
        )
        best_individuals = [crystal for crystal, _ in sorted_list[:self.n]]

        return best_individuals

    def select_start_pop(
            self, individuals: list[Individual]
    ) -> list[Individual]:
        """
        Returns n crystals with the highest fitness.
        """

        if self.n > len(individuals):
            n = len(individuals)
            warnings.warn(
                "BestCrystalsPopulation: " +
                "n is larger than the number of individuals. " +
                "Returning {} instead of {} individuals.".format(
                    n, self.n
                )
            )
            return individuals

        chosen_individuals = self.__get_n_best_individuals(
            individuals=individuals,
            verbose=self.verbose
        )
        return chosen_individuals

    def __repr__(self) -> str:
        if self.database_name is not None:
            return "BestCrystalsPopulation(n={}, database_name={})".format(
                self.n, self.database_name
            )
        else:
            return "BestCrystalsPopulation(n={})".format(self.n)


class WorstCrystalsPopulation(PopulationSelection):
    """
    A class that returns a list of n crystals with the highest fitness.
    Set database_name to describe the database the original crystals are from.

    Call the object after initialization to get the list of crystals.
    """

    def __init__(
        self,
        n: int,
        evaluation_function: Callable[[Individual], float],
        database_name: Optional[str] = None,
        verbose: int = 1
    ):
        self.n = n
        self.evaluation_function = evaluation_function
        self.database_name = database_name
        self.verbose = verbose

    def __get_n_worst_individuals(
        self,
        individuals: list[Individual],
        verbose: int = 1
    ) -> list[Individual]:
        fitness_list = []
        for i, crystal in enumerate(individuals):
            fitness_list.append(
                self.evaluation_function(crystal)
            )

        sorted_list = sorted(
            zip(individuals, fitness_list),
            key=lambda x: x[1],
            reverse=False
        )
        sorted_crystals = [crystal for crystal, _ in sorted_list]

        return sorted_crystals[:self.n]

    def select_start_pop(
        self, individuals: list[Individual]
    ) -> list[Individual]:
        """
        Returns n crystals with the highest fitness.
        """
        chosen_individuals = self.__get_n_worst_individuals(
            individuals=individuals,
            verbose=self.verbose
        )
        return chosen_individuals

    def __repr__(self) -> str:
        if self.database_name is not None:
            return "WorstCrystalsPopulation(n={}, database_name={})".format(
                self.n, self.database_name
            )
        else:
            return "WorstCrystalsPopulation(n={}, ".format(
                self.n
            )


class SelectAllPopulation(PopulationSelection):
    """
    A class that returns all crystals.
    Set database_name to describe the database the original crystals are from.

    Call the object after initialization to get the list of crystals.
    """

    def __init__(
        self,
        database_name: Optional[str] = None
    ):
        self.database_name = database_name

    def select_start_pop(
        self,
        individuals: list[Individual],
        n: int
    ) -> list[Individual]:
        """Returns all crystals. Parameter n is ignored."""
        return individuals

    def __repr__(self) -> str:
        if self.database_name is not None:
            return "SelectAllPopulation(database_name={})".format(
                self.database_name
            )
        else:
            return "SelectAllPopulation()"


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
        sort_by: Callable[[Individual], float] | None = lambda x: x.fitness.values[0],
    ):
        super().__init__()
        self.sort_by = sort_by

    def select(
        self, individuals: list[Individual], n: int
    ) -> list[Individual]:
        # Assign crowding distance to each individual, as expected by the
        # selTournamentDCD function
        tools.emo.assignCrowdingDist(individuals)

        # Select individuals using the selTournamentDCD function
        selected_individuals = tools.selTournamentDCD(individuals, n)

        # Reset the crowding distance of the selected individuals
        for individual in selected_individuals:
            if hasattr(individual, "crowding_dist"):
                del individual.crowding_dist

        if self.sort_by is not None:
            selected_individuals.sort(
                key=self.sort_by,
            )

        return selected_individuals

    def __repr__(self) -> str:
        return "TournamentDCDSelection()"

class DopePopulationSelection(PopulationSelection):
    def __init__(
        self,
        atom_types: list[str],
        cell_bounds: CustomCellBounds,
        generator: PopulationGenerator,
    ) -> None:
        self.atom_types = atom_types
        self.cell_bounds = cell_bounds
        self.generator = generator

    def select(
        self, individuals: list[Individual], n: int
    ) -> list[Individual]:
        """Add randomly generated crystals to the population.

        :param individuals: A list of individuals to add to.
        :param n: The number of individuals to add to the population.

        :return: A list of individuals with n new individuals added.
            Therefore the length of the list is len(individuals) + n.
        """
        new_individuals = self.generator.generate_individuals(n)

        return individuals + new_individuals

    def __repr__(self) -> str:
        return "DopePopulationSelection(atom_types={}, cell_bounds={}, generator={})".format(
            self.atom_types, self.cell_bounds
        )
