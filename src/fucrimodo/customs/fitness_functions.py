import numpy as np
from ase.geometry import get_distances
from sklearn.metrics.pairwise import rbf_kernel

from ..core import Individual
from ..core.abstracts import FitnessFunction
from ..core.utils import CustomClosestDistances
from .global_soap_target import GlobalSOAP


class PhysicalityFitness(FitnessFunction):
    """Fitness that rewards physically plausible atomic structures.

    Penalizes pairwise interatomic distances that are shorter than the minimum
    allowed distances for the corresponding element pair. The penalty is zero
    when all distances are above their thresholds; it grows as atoms get closer
    than allowed. The final fitness is computed as the exponential of the
    negative total penalty, yielding a value in the range ``(0, 1]``.

    :param closest_distances: Object providing the minimum allowed distance
        for each pair of atomic numbers. Idea: Use a stricter closest
        distances here than for the mutations/crossovers, this way structures
        with too close atoms can be generated but punished. This way unexpected
        new materials can be found.  (`Fishes with faces who come out of the sea
        cause tsunamis.` ~ Toki)
    :param db_title: Optional title for the structures database.
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        db_title: str | None = "PhysicalityFitness",
    ):
        super().__init__(db_title=db_title)
        self.closest_distances = closest_distances

    def evaluate_individual(self, individual: Individual) -> float:
        positions = individual.get_positions()
        atomic_numbers = individual.get_atomic_numbers()
        cell = individual.get_cell()

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

    def __repr__(self) -> str:
        r_str = "PhysicalityFitness()"
        return r_str


# Utility for fitness functions that use the similarity
def _assign_features_to_individuals(
    soap_obj: GlobalSOAP, individuals: list[Individual], n_jobs
):
    """Calculates and assigns features to the individuals if they are not already set."""
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
    """Fitness based on RBF similarity to target SOAP features.

    Evaluates individuals by comparing their SOAP feature vectors to the target
    SOAP features using the radial basis function (RBF) kernel. Higher
    similarity to the target yields a higher fitness value.  Note: During the
    evaluation the features stored in the :attr:`Individual.features` will be
    used if present. If not present they are calculated and assigned with the
    :attr:`soap_obj`.

    :param target_soap_features: Target SOAP feature vector to compare
        against.
    :param soap_object: :class:`GlobalSOAP` instance used to compute SOAP
        feature vectors for individuals that do not already have features
        assigned.
    :param rbf_gamma: Gamma parameter for the RBF kernel. Controls the
        width of the Gaussian similarity function. Default is ``0.1``.
    :param db_title: Optional title for the structures database.
    :param round_result: Optional number of decimal places to round the
        fitness values to. If ``None``, no rounding is applied.
    :param n_jobs: Number of parallel jobs to use when computing SOAP
        features for individuals. Default is ``1``.
    """

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

        if self.round_result:
            return [round(f, self.round_result) for f in fitnesses]
        else:
            return fitnesses.tolist()

    def __repr__(self) -> str:
        r_str = "SoapRbfSimilarityFitness("
        r_str += f"rbf_gamma={self.rbf_gamma}"
        r_str += ")"
        return r_str


class SpeciesSpecificSoapRbfSimFitness(FitnessFunction):
    """Fitness based on species-specific RBF similarity to target SOAP features.

    Evaluates individuals by comparing only the species-pair-specific segment of
    their SOAP feature vectors to the corresponding segment of a target SOAP
    feature vector. The comparison is performed using the radial basis function
    (RBF) kernel. Higher similarity to the target yields a higher fitness value.

    The relevant segment of the SOAP vector is determined by the
    :attr:`species` pair and extracted via the :meth:`GlobalSOAP.get_location`
    method of the provided SOAP object.

    :param target_soap_features: Target SOAP feature vector to compare against.
    :param soap_object: :class:`GlobalSOAP` instance used to compute SOAP
        feature vectors for individuals that do not already have features
        assigned, and to locate the species-specific slice of the SOAP vector.
    :param species: Tuple of two chemical species whose SOAP interaction block
        should be used for the similarity calculation.
    :param rbf_gamma: Gamma parameter for the RBF kernel. Controls the
        width of the Gaussian similarity function. Default is ``0.1``.
    :param db_title: Optional title for the structures database.
    :param round_result: Optional number of decimal places to round the
        fitness values to. If ``None``, no rounding is applied.
    :param n_jobs: Number of parallel jobs to use when computing SOAP
        features for individuals. Default is ``1``.
    """

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

        if self.round_result:
            return [round(f, self.round_result) for f in fitnesses]
        else:
            return fitnesses.tolist()

    def __repr__(self) -> str:
        r_str = "SpeciesSpecificSoapRbfSimFitness("
        r_str += f"rbf_gamma={self.rbf_gamma}"
        r_str += f"species={self.rbf_gamma}"
        r_str += ")"
        return r_str
