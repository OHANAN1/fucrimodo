import numpy as np
import ase
from ase.geometry import get_distances
from sklearn.metrics.pairwise import rbf_kernel
from .global_soap_target import GlobalSOAP
from ..core.abstracts import FitnessFunction
from ..core.utils import CustomClosestDistances
from ..core import Individual


class PhysicalityFitness(FitnessFunction):
    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        db_title: str | None = "PhysicalityFitness",
    ):
        super().__init__(db_title=db_title)
        self.closest_distances = closest_distances

    def __calculate_normalized_atom_distance_fitness(
        self,
        structure: ase.Atoms,
    ) -> float:
        """
        The Bigger the better.
        Calculates the distances of all atoms in the structure.
        If the distance between two atoms is bigger or equal to the
        min_allowed_dist the fitness is increased by 1.

        The minimal distance between two atoms is calculated by the
        covalent radii of the atoms with the closest_distances_generator
        function from ase_ga.utilities.

        Nomalized by N(N-1)/2.
        """
        positions = structure.get_positions()
        atomic_numbers = structure.get_atomic_numbers()
        cell = structure.get_cell()

        _, distances = get_distances(p1=positions, cell=cell, pbc=True)

        exponent = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = distances[i, j]
                min_allowed_dist = self.closest_distances[
                    (atomic_numbers[i], atomic_numbers[j])
                ]
                exponent += np.max(
                    [(min_allowed_dist - distance) / min_allowed_dist, 0],
                )

        fitness = np.exp(-exponent)

        return fitness

    def evaluate_individual(self, individual: Individual) -> float:
        return self.__calculate_normalized_atom_distance_fitness(
            structure=individual,
        )

    def __repr__(self) -> str:
        r_str = "PhysicalityFitness()"
        return r_str


# Utilitie for fitness functions that use the similarity
def _assign_features_to_individuals(
    soap_obj: GlobalSOAP, individuals: list[Individual], n_jobs
):
    """Assigns the features to the individuals if they are not already set."""
    # Collect individuals without features
    individuals_without_features = []
    for ind in individuals:
        if ind.features is None:
            individuals_without_features.append(ind)

    if len(individuals_without_features) > 0:
        feature_vectors = soap_obj.create(individuals_without_features, n_jobs=n_jobs)

        for i, feature_vector in enumerate(feature_vectors):
            individuals_without_features[i].features = feature_vector


class SoapRbfSimilarityFitness(FitnessFunction):
    def __init__(
        self,
        target_soap_features: np.ndarray,
        soap_object: GlobalSOAP,
        rbf_gamma: float = 0.1,
        db_title: str | None = None,
        round_result: int | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(db_title=db_title)
        self.target_soap_features = target_soap_features
        self.soap_obj = soap_object
        self.round_result = round_result
        self.rbf_gamma = rbf_gamma
        self.n_jobs = n_jobs

    def _get_similarities_to_target(
        self,
        feature_vector_list: list[np.ndarray],
    ) -> np.ndarray:
        similarity_matrix = rbf_kernel(
            feature_vector_list, [self.target_soap_features], gamma=self.rbf_gamma
        )
        return similarity_matrix.flatten()

    def evaluate_individual(self, individual: Individual) -> float:
        return self.evaluate_individuals([individual])[0]

    def evaluate_individuals(self, individuals: list[Individual]) -> list[float]:
        """Evaluate a similarity fitness for a list of individuals.

        Uses the :attr:`Individual.features` attribute to calculate the fitness.
        If not set calculates the features with the :attr:`GlobalSOAP` object
        for all individuals without features in parallel.

        :param individuals: List of individuals to evaluate.

        :returns: List of fitness values for each individual.

        :raises ValueError: If the features could not be assigned to the
            individuals or the fitness could not be calculated.
        """
        # Assign the features to the individuals if they are not set already
        _assign_features_to_individuals(
            soap_obj=self.soap_obj,
            individuals=individuals,
            n_jobs=self.n_jobs,
        )

        # Calculate the fitness for each individual
        features = []
        for ind in individuals:
            # Check again if the features are set
            assert (
                type(ind.features) is np.ndarray
            ), "Features are not set. This should not happen."
            features.append(ind.features)

        # Use the feature vector to calculate the fitnesses/similarities
        fitnesses = self._get_similarities_to_target(feature_vector_list=features)

        if self.round_result is not None:
            return [round(f, self.round_result) for f in fitnesses]
        else:
            return fitnesses.tolist()

    def __repr__(self) -> str:
        r_str = "SoapRbfSimilarityFitness("
        r_str += f"rbf_gamma={self.rbf_gamma}"
        r_str += ")"
        return r_str


class SpeciesSpecificSoapRbfSimFitness(FitnessFunction):
    def __init__(
        self,
        target_soap_features: np.ndarray,
        soap_object: GlobalSOAP,
        species: tuple[str, str],
        rbf_gamma: float = 0.1,
        db_title: str | None = None,
        round_result: int | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(db_title=db_title)
        self.target_soap_features = target_soap_features
        self.soap_obj = soap_object
        self.round_result = round_result
        self.rbf_gamma = rbf_gamma
        self.n_jobs = n_jobs
        self.species = species

    def _get_rbf_sim_for_species(
        self,
        feature_vector_list: list[np.ndarray],
    ) -> np.ndarray:
        species_slice = self.soap_obj.get_location(self.species)

        similarity_matrix = rbf_kernel(
            [v[species_slice] for v in feature_vector_list],
            [self.target_soap_features[species_slice]],
            gamma=self.rbf_gamma,
        )
        return similarity_matrix.flatten()

    def evaluate_individual(self, individual: Individual) -> float:
        return self.evaluate_individuals([individual])[0]

    def evaluate_individuals(self, individuals: list[Individual]) -> list[float]:
        """Evaluate a similarity fitness for a list of individuals.

        Uses the :attr:`Individual.features` attribute to calculate the fitness.
        If not set calculates the features with the :attr:`GlobalSOAP` object
        for all individuals without features in parallel.

        :param individuals: List of individuals to evaluate.

        :returns: List of fitness values for each individual.

        :raises ValueError: If the features could not be assigned to the
            individuals or the fitness could not be calculated.
        """
        # Assign the features to the individuals if they are not set already
        _assign_features_to_individuals(
            soap_obj=self.soap_obj,
            individuals=individuals,
            n_jobs=self.n_jobs,
        )

        # Calculate the fitness for each individual
        features = []
        for ind in individuals:
            # Check again if the features are set
            assert (
                type(ind.features) is np.ndarray
            ), "Features are not set. This should not happen."
            features.append(ind.features)

        # Use the feature vector to calculate the fitnesses/similarities
        fitnesses = self._get_rbf_sim_for_species(feature_vector_list=features)

        if self.round_result is not None:
            return [round(f, self.round_result) for f in fitnesses]
        else:
            return fitnesses.tolist()

    def __repr__(self) -> str:
        r_str = "SpeciesSpecificSoapRbfSimFitness("
        r_str += f"rbf_gamma={self.rbf_gamma}"
        r_str += f"species={self.rbf_gamma}"
        r_str += ")"
        return r_str
