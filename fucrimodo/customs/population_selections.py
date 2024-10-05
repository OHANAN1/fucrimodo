import random
from typing import Callable, Optional, Sequence
from fucrimodo.core.modules.individual import Individual
from fucrimodo.core.utils.custom_soap import CustomSOAP
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import rbf_kernel
import warnings
from fucrimodo.core.modules import PopulationSelection
from deap import tools

# TODO: Needs work when changing PopulationGenerationClass
import fucrimodo.customs.population_generator as crystal_creation

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
        # Perform NSGA-II selection.
        individuals = tools.selTournament(
            individuals, k=n, tournsize=self._tournament_size
        )

        # Ensure that the number of individuals selected is not bigger n.
        if len(individuals) > n:
            individuals = individuals[:n]

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

        # Ensure that the number of individuals selected is not bigger n.
        if len(individuals) > n:
            individuals = individuals[:n]

        return individuals

# class DiversityPopulation(PopulationSelection):
#     """
#     A class that returns a list of n crystals with the highest diversity.
#     Set database_name to describe the database the original crystals are from.
#
#     Call the object after initialization to get the list of crystals.
#     """
#
#     def __init__(
#         self,
#         soap_object: CustomSOAP,
#         n: int,
#         gamma: float | None = None,
#         verbose: int = 1
#     ):
#         self.n = n
#         self.soap_object = soap_object
#         self.gamma = gamma
#         self.verbose = verbose
#
#     def __get_similarity_to_others(
#         self,
#         soap_feature_vectors: Sequence[NDArray[np.float64]],
#         gamma: float | None = None
#     ) -> NDArray[np.float64]:
#         """
#         This function should return a list of floats that
#         represent the fitness of the crystals based on their
#         similarity to the other crystals. The similarity
#         is calculated with the SOAP kernel.
#         """
#         similarity_matrix = rbf_kernel(
#             soap_feature_vectors,
#             gamma=gamma
#         )
#
#         mean_similarities = np.array([])
#         n_features = len(soap_feature_vectors)
#         for i in range(n_features):
#             similarity = 0
#             for j in range(n_features):
#                 if i != j:  # avoid counting fitness to itselve
#                     similarity += similarity_matrix[i][j]
#
#             mean_similarities = np.append(
#                 mean_similarities, similarity/n_features)
#         return mean_similarities
#
#     def __get_n_diverse_indivduals(
#         self,
#         individuals: list[Individual],
#         n: int,
#         verbose: int = 1
#     ) -> list[Individual]:
#
#         soap_feature_vectors = self.soap_object.create(individuals)
#
#         similarities_to_others = self.__get_similarity_to_others(
#             soap_feature_vectors=soap_feature_vectors,  # type: ignore
#             gamma=self.gamma
#         )
#
#         if verbose >= 1:
#             print("Sorting crystals by diversity...")
#             print()
#
#         sorted_list = sorted(
#             zip(individuals, similarities_to_others),
#             key=lambda x: x[1],
#             reverse=True
#         )
#         sorted_individuals = [crystal for crystal, _ in sorted_list]
#
#         diverse_individuals = sorted_individuals[:n]
#
#         return diverse_individuals
#
#     def select_start_pop(
#         self,
#         individuals: list[Individual]
#     ) -> list[Individual]:
#         """
#         Returns n crystals with the highest diversity.
#         """
#         if self.n > len(individuals):
#             n = len(individuals)
#             warnings.warn(
#                 "DiversityPopulation: " +
#                 "n is larger than the number of individuals. " +
#                 "Returning {} instead of {} individuals.".format(
#                     n, self.n
#                 )
#             )
#             return individuals
#
#         chosen_individuals = self.__get_n_diverse_indivduals(
#             individuals=individuals,
#             n=self.n,
#         )
#         return chosen_individuals
#
#     def __repr__(self) -> str:
#         return "DiversityPopulation(n={}, gamma={})".format(
#             self.n, self.gamma
#         )


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


class DopePopulationSelection(PopulationSelection):

    def __init__(
        self, atom_types: list[str], cell_bounds: CustomCellBounds
    ) -> None:
        self.atom_types = atom_types
        self.cell_bounds = cell_bounds

    def select(
        self, individuals: list[Individual], n: int
    ) -> list[Individual]:
        """Add randomly generated crystals to the population.

        :param individuals: A list of individuals to add to.
        :param n: The number of individuals to add to the population.

        :return: A list of individuals with n new individuals added.
            Therefore the length of the list is len(individuals) + n.
        """
        new_crystal = crystal_creation.create_one_atomic_crystals(
            atom_types=self.atom_types,
            cell_bounds=self.cell_bounds,
            total_number_of_atoms=n
        )

        new_individuals = [
            Individual(
                positions=crystal.positions,
                cell=crystal.cell,
                pbc=crystal.pbc,
                symbols=crystal.get_chemical_symbols(),
            ) for crystal in new_crystal
        ]

        return individuals + new_individuals

    def __repr__(self) -> str:
        return "DopePopulationSelection(atom_types={}, cell_bounds={})".format(
            self.atom_types, self.cell_bounds
        )


class ReplaceAndReaddLater(PopulationSelection):

    def __init__(
        self, atom_types: list[str], cell_bounds: CustomCellBounds, after_n: int
    ) -> None:
        self.atom_types = atom_types
        self.cell_bounds = cell_bounds
        self.after_n = after_n

        self.current_n = 0

        self.old_individuals = []

    def select_start_pop(
        self,
        individuals: list[Individual]
    ) -> list[Individual]:
        self.old_individuals += individuals
        self.current_n += 1

        if self.current_n < self.after_n:
            new_individuals = crystal_creation.create_one_atomic_crystals(
                atom_types=self.atom_types,
                cell_bounds=self.cell_bounds,
                total_number_of_atoms=len(individuals),
            )
            return new_individuals

        else:
            return self.old_individuals


    def __repr__(self) -> str:
        return "ReplaceAndReaddLater(after_n={}, atom_types={}, cell_bounds={})".format(
            self.after_n, self.atom_types, self.cell_bounds
        )
