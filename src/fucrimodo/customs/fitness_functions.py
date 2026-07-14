import numpy as np
import ase
from ase.geometry import get_distances
import warnings
from fucrimodo.customs.global_soap_target import GlobalSOAP
from fucrimodo.core.modules import FitnessFunction, Individual
from fucrimodo.utils.soap_similarity import SOAPSimilarity
import datetime


class PhysicalityFitness(FitnessFunction):
    def __init__(
        self,
        closest_distances: dict[tuple[int, int], float],
        db_title: str | None = "PhysicalityFitness",
    ):
        super().__init__(db_title=db_title)
        self.closest_distances = closest_distances

    def __calculate_normalized_atom_distance_fitness(
        self,
        crystal: ase.Atoms,
    ) -> float:
        """
        The Bigger the better.
        Calculates the distances of all atoms in the crystal.
        If the distance between two atoms is bigger or equal to the
        min_allowed_dist the fitness is increased by 1.

        The minimal distance between two atoms is calculated by the
        covalent radii of the atoms with the closest_distances_generator
        function from ase.ga.utilities.

        Nomalized by N(N-1)/2.
        """
        positions = crystal.get_positions()
        atomic_numbers = crystal.get_atomic_numbers()
        cell = crystal.get_cell()

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
            crystal=individual,
        )

    def __repr__(self) -> str:
        r_str = "PhysicalityFitness()"
        return r_str


class SimilarityToTargetSOAPFitness(FitnessFunction):
    def __init__(
        self,
        target_soap_features,
        soap_object: GlobalSOAP,
        soap_similarity: SOAPSimilarity,
        adjust: bool = False,
        db_title: str | None = None,
        round_result: int | None = None,
        n_jobs: int = 1,
    ):
        super().__init__(db_title=db_title)
        self.target_soap_features = target_soap_features
        self.soap_similarity = soap_similarity
        self.soap_obj = soap_object
        self.adjust = adjust
        self.round_result = round_result
        self.n_jobs = n_jobs

    def __assign_features_to_individuals(self, individuals: list[Individual]):
        """Assigns the features to the individuals if they are not already set."""

        # Collect individuals without features
        individuals_without_features = []
        for ind in individuals:
            if ind.features is None:
                individuals_without_features.append(ind)

        if len(individuals_without_features) > 0:
            feature_vectors = self.soap_obj.create(
                individuals_without_features, n_jobs=self.n_jobs
            )

            # If only one individual is without features don't loop
            if len(individuals_without_features) == 1:
                individuals_without_features[0].features = feature_vectors
            else:
                # Else loop through the feature vectors and assign them to the
                # individuals
                for i, feature_vector in enumerate(feature_vectors):
                    individuals_without_features[i].features = feature_vector

    def evaluate_individual(self, individual: Individual) -> float:
        try:
            # Assign the features to the individual if they are not set already
            self.__assign_features_to_individuals([individual])

            # Check again if the features are set
            assert (
                individual.features is not None
            ), "Features are not set. This should not happen."

        except Exception as e:
            warnings.warn(
                f"{self.db_title}:"
                f"Could not use soap_create for ind: {individual}"
                f"Error: {e}"
            )
            return 0

        try:
            # Calculate the fitness for the individual, here the similarity
            similarity = self.soap_similarity.get_similarity_of_feature_vector(
                feature_vector=individual.features,
            )
            if self.round_result is not None:
                return round(similarity, self.round_result)
            else:
                return similarity

        except Exception as e:
            warnings.warn(
                f"{self.db_title}:"
                f"Could not calculate fitness for ind: {individual}\n"
                f"Error: {e} \n"
                f"Shape of feature vector: {individual.features.shape}"
            )
            return 0

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
        # Use the evaluate_individual function if only one individual is given
        if len(individuals) == 1:
            return [self.evaluate_individual(individuals[0])]

        # Else calculate the fitness for all individuals
        try:
            # Assign the features to the individuals if they are not set already
            self.__assign_features_to_individuals(individuals)
        except Exception as e:
            raise ValueError(
                f"{self.db_title}:"
                f"Could not assign features to indviduals: {individuals}"
                f"Error: {e}"
            )

        try:
            # Calculate the fitness for each individual
            features = []
            for ind in individuals:
                # Check again if the features are set
                assert (
                    ind.features is not None
                ), "Features are not set. This should not happen."
                features.append(ind.features)

            # Use the feature vector to calculate the fitnesses/similarities
            fitnesses = self.soap_similarity.get_similarity_of_feature_vectors(
                feature_vectors=features
            )
            if self.round_result is not None:
                return [round(f, self.round_result) for f in fitnesses]
            else:
                return fitnesses.tolist()

        except Exception as e:
            raise ValueError(
                f"{self.db_title}:"
                f"Could not calculate fitness for individuals: {individuals}\n"
                f"Error: {e}"
            )

    def __repr__(self) -> str:
        r_str = "SimilarityToTargetSOAPFitness("
        r_str += f"soap_similarity={self.soap_similarity.__repr__()}"
        r_str += ")"
        return r_str


class VolumeFitness(FitnessFunction):
    """Decreases the fitness with the volume of the individual.

    The fitness is calculated as: fitness = np.exp(-self.gamma * volume)
    The bigger the volume the smaller the fitness.

    :param gamma: Scaling factor for the volume.
    :param db_title: Title of the database.
    :param round_volume: Round the volume to this number of decimals to
        avoid prefering only slightly smaller volumes.
    """

    def __init__(
        self, db_title: str | None = None, gamma: float = 0.01, round_volume: int = 1
    ):
        super().__init__(db_title=db_title)
        self.gamma = gamma
        self.round_volume = round_volume

    def evaluate_individual(self, individual: ase.Atoms) -> float:
        try:
            volume = round(individual.get_volume(), self.round_volume)
            return np.exp(-self.gamma * volume)
        except Exception as e:
            warnings.warn(
                f"{self.db_title}: "
                f"Could not calculate fitness for ind: {individual}\n"
                f"Error: {e}"
            )
            return 0


class NumberOfAtomsFitness(FitnessFunction):
    """
    Uses the arctan function to scale the fitness.
    n_max is the max number of atoms one individual should have.
    The fitness is scaled from 0 to 1. The more atoms the smaller the fitness.

    fitness = (np.arctan(-n_atoms + self.n_max + 3)+(np.pi / 2)) / np.pi
    """

    def __init__(self, n_max: int = 10, db_title: str | None = None):
        super().__init__(db_title=db_title)
        self.n_max = n_max

    def evaluate_individual(self, individual: ase.Atoms) -> float:
        try:
            n_atoms = len(individual)
            fitness = (np.arctan(-n_atoms + self.n_max + 3) + (np.pi / 2)) / np.pi
            return fitness
        except Exception as e:
            warnings.warn(
                f"{self.db_title}: "
                f"Could not calculate fitness for ind: {individual}\n"
                f"Error: {e}"
            )
            return 0

    def __repr__(self) -> str:
        r_str = "NumberOfAtomsFitness()"
        return r_str


class AgeFitness(FitnessFunction):
    """Decreases the fitness with the age of the individual relative to the init time.

    The fitness is calculated as: fitness = (1 - np.exp(- self.gamma * relative_age_seconds))
    Where the relative age is the time between the creation time of the individual
    and the init time of the fitness function.

    Does not calculate the age of an individual directly, since the fitness
    values are only calculated during the creation of an individual.
    With this approach the age of different individuals can be compared
    relative to the init time of the fitness function.
    Please avoid creation times smaller than the init time of the fitness
    function.

    Very different from, but inspired by:

    Wenhui Yang, Edirisuriya M. Dilanga Siriwardane, Jianjun Hu;
    Doi: doi.org/10.48550/arXiv.2107.01346

    :param gamma: Scaling factor for the age.
    :param db_title: Title of the database.
    :param round_fitness: Round the fitness to this number of decimals
        to avoid prefering structures that where created only slightly
        earlier due to unrelated timing of structure creation.
    """

    def __init__(
        self,
        gamma: float = 0.001,
        db_title: str = "AgeFitness",
        round_fitness: int = 15,
    ):
        super().__init__(db_title=db_title)
        self.gamma = gamma
        self.init_time = datetime.datetime.now()
        self.round_fitness = round_fitness

    def evaluate_individual(self, individual: Individual) -> float:
        try:
            # Calculate the age relative to the init time
            relative_age = individual.creation_time - self.init_time

            # Convert the datetime object to integer seconds
            relative_age_seconds = int(relative_age.total_seconds())

            # Avoid division by zero
            # Avoid negative values that would drastically increase the fitness
            if relative_age_seconds <= 0:
                return 0.0

            else:
                # Add + 0.0 to make sure the fitness is a float
                fitness = (1 - np.exp(-self.gamma * relative_age_seconds)) + 0.0
                return round(fitness, self.round_fitness)

        except Exception as e:
            warnings.warn(
                f"{self.db_title}: "
                f"Could not calculate fitness for ind: {individual}\n"
                f"Error: {e}"
            )
            return 0


class DummyFitness(FitnessFunction):
    def __init__(self):
        pass

    def evaluate_individual(self, individual: ase.Atoms) -> float:
        return 999.0

    def __repr__(self):
        return "DummyFitness"
