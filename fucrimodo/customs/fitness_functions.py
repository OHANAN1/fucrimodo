import numpy as np
import ase
from fucrimodo.core.utils.custom_soap import CustomSOAP
from numpy.typing import NDArray
from ase.geometry import get_distances
import warnings
from fucrimodo.core.modules import FitnessFunction
from fucrimodo.utils.soap_similarity import SOAPSimilarity


# ╔══════════════════════════════════════════════════════════╗
# ║                    Utility functions                     ║
# ╚══════════════════════════════════════════════════════════╝


# ╔══════════════════════════════════════════════════════════╗
# ║                 Fitness Function Classes                 ║
# ╚══════════════════════════════════════════════════════════╝

class PhysicalityFitness(FitnessFunction):
    def __init__(
        self,
        closest_distances: dict[tuple[int, int], float],
        db_title: str | None = "PhysicalityFitness"
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

        d_matrix, distances = get_distances(
            p1=positions,
            cell=cell,
            pbc=True
        )

        fitness = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = distances[i, j]
                min_allowed_dist = self.closest_distances[
                    (atomic_numbers[i], atomic_numbers[j])
                ]
                if distance >= min_allowed_dist:
                    fitness += 1

        if len(atomic_numbers) > 1:
            norm_factor = 2/(len(atomic_numbers) * (len(atomic_numbers) - 1))
        else:
            norm_factor = 1

        return fitness * norm_factor

    def evaluate_individual(self, individual: ase.Atoms) -> float:
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
        soap_object: CustomSOAP,
        soap_similarity: SOAPSimilarity,
        adjust: bool = False,
        db_title: str | None = None
    ):
        super().__init__(db_title=db_title)
        self.target_soap_features = target_soap_features
        self.soap_similarity = soap_similarity
        self.soap_obj = soap_object
        self.adjust = adjust

    def __get_similarity_to_target_soap(
        self,
        soap_feature_vector: NDArray[np.float64]
    ) -> float:
        similarity = self.soap_similarity.get_similarity_of_feature_vector(
            soap_feature_vector
        )
        return similarity

    def __get_difference_to_target_fitness(
        self,
        soap_feature_vector: NDArray[np.float64],
    ) -> float:
        similarities_to_target_soap = self.__get_similarity_to_target_soap(
            soap_feature_vector
        )
        return similarities_to_target_soap

    def evaluate_individual(self, individual: ase.Atoms) -> float:

        try:
            if hasattr(individual, "soap_feature_vector"):
                soap_feature_vector = individual.soap_feature_vector # type: ignore

            else:
                soap_feature_vector = self.soap_obj.create(individual)
                individual.soap_feature_vector = soap_feature_vector # type: ignore


        except Exception as e:
            warnings.warn(
                f"{self.db_title}:"
                f"Could not use soap_create for ind: {individual}"
                f"Error: {e}"
            )
            return 0

        try:
            diff_to_target_fitnesses = self.__get_difference_to_target_fitness(
                soap_feature_vector=soap_feature_vector
            )
            return diff_to_target_fitnesses

        except Exception as e:
            warnings.warn(
                f"{self.db_title}:"
                f"Could not calculate fitness for ind: {individual}\n"
                f"Error: {e} \n"
                f"Shape of feature vector: {soap_feature_vector.shape}"
            )
            return 0

    def __repr__(self) -> str:
        r_str = "SimilarityToTargetSOAPFitness("
        r_str += f"soap_similarity={self.soap_similarity.__repr__()}"
        r_str += ")"
        return r_str


class NumberOfAtomsFitness(FitnessFunction):
    """
    Uses the arctan function to scale the fitness.
    n_max is the max number of atoms one individual should have.
    The fitness is scaled from 0 to 1. The more atoms the smaller the fitness.

    fitness = (np.arctan(-n_atoms + self.n_max + 3)+(np.pi / 2)) / np.pi
    """

    def __init__(
        self,
        n_max: int = 10,
        db_title: str | None = None
    ):
        super().__init__(db_title=db_title)
        self.n_max = n_max

    def evaluate_individual(self, individual: ase.Atoms) -> float:
        try:
            n_atoms = len(individual)
            fitness = (np.arctan(-n_atoms + self.n_max + 3)+(np.pi / 2)) / np.pi
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


class DummyFitness(FitnessFunction):
    def __init__(self):
        pass

    def evaluate_individual(self, individual: ase.Atoms) -> float:
        return 999.

    def __repr__(self):
        return "DummyFitness"
