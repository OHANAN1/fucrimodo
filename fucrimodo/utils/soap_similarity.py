from abc import ABC, abstractmethod
from typing import Sequence
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from numpy.typing import NDArray
import numpy as np
from fucrimodo.core.utils.custom_soap import CustomSOAP
from dscribe.kernels import AverageKernel
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings


class SOAPSimilarity(ABC):
    def __init__(self, target_feature_vector: NDArray[np.float64]) -> None:
        self.target_feature_vector = target_feature_vector

    @abstractmethod
    def get_similarity_of_feature_vector(
        self, feature_vector: NDArray[np.float64]
    ) -> float:
        pass

    @abstractmethod
    def get_similarity_of_feature_vectors(
        self,
        feature_vectors: Sequence[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def adjust_to_population(self, population: Sequence[NDArray[np.float64]]):
        """
        Adjust the similarity function to the population.
        Can be used for example to adjust the gamma value of an RBF kernel.
        If not needed, this function can be left empty.
        """
        pass

    @abstractmethod
    def set_db_title(self, title: str) -> None:
        """
        Optional function that can be used to set the title of the
        ase database. Therefore no space, no numbers and no special
        characters are allowed. '_' is allowed.
        """
        forbidden_chars = [
            " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
        ]
        if any([char in title for char in forbidden_chars]):
            raise ValueError(
                "The title of the database must not contain any spaces, "
                "numbers or special characters."
                f"Given title: {title}"
            )

        self.db_title = title

    @abstractmethod
    def get_db_title(self) -> str:
        """
        Optional function that can be used to get the title of the
        ase database.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

# ╔══════════════════════════════════════════════════════════╗
# ║                    Utility functions                     ║
# ╚══════════════════════════════════════════════════════════╝


def calculate_variance_for_gamma(
    soap_feature_vectors: Sequence[NDArray[np.float64]],
    target_soap_feature_vector: NDArray[np.float64],
    gamma: float
) -> float:
    similarities = rbf_kernel(
        soap_feature_vectors,
        [target_soap_feature_vector],
        gamma=gamma
    )
    variance = np.var(similarities, axis=0).mean()
    return variance


def find_best_gamma_for_target_comparison(
    soap_feature_vectors: Sequence[NDArray[np.float64]],
    target_soap_feature_vector: NDArray[np.float64],
    gamma_values: NDArray[np.float64],
) -> float:
    print("Finding best gamma...")
    best_gamma = None
    highest_variance = -np.inf  # Verwenden von -np.inf für den Anfangswert
    variances = []

    for gamma in gamma_values:
        variance = calculate_variance_for_gamma(
            soap_feature_vectors,
            target_soap_feature_vector,
            gamma
        )
        variances.append(variance)
        if variance > highest_variance:
            highest_variance = variance
            best_gamma = gamma

    if best_gamma is None:
        raise ValueError("Best gamma was not found")

    print(f"Best gamma: {best_gamma:.7f}, with variance: {highest_variance}")
    print()

    return best_gamma


def find_best_gamma_for_rbf_kernel(
    soap_feature_vectors: Sequence[NDArray[np.float64]],
    gamma_values: NDArray[np.float64],
    plot: bool = False
) -> float:
    print()
    print("Finding best gamma...")
    best_gamma = None
    highest_variance = 0
    variances = []
    for gamma in gamma_values:
        print(f"Calculating variance for gamma {gamma:.3f}", end="\r")
        similarities = rbf_kernel(
            soap_feature_vectors,
            soap_feature_vectors,
            gamma=gamma
        )
        variance = np.var(similarities, axis=0).mean()
        variances.append(variance)
        if variance > highest_variance:
            best_gamma = gamma
            highest_variance = variance

    print()

    if best_gamma is None:
        raise ValueError("Gamma was not found")

    print(f"Best gamma: {best_gamma:.3f}")

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(gamma_values, variances)
        ax.set_title("Variance of Similarities")
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Variance")
        ax.set_xscale("log")

    return best_gamma


# ╔══════════════════════════════════════════════════════════╗
# ║                   Similarity Classes                     ║
# ╚══════════════════════════════════════════════════════════╝

class CosineSimilarity(SOAPSimilarity):
    def __init__(self, target_feature_vector: NDArray[np.float64]) -> None:
        self.target_feature_vector = target_feature_vector

    def __get_cosine_similarity_to_target(
        self,
        feature_vector_list: Sequence[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        similarity_matrix = cosine_similarity(
            feature_vector_list,
            [self.target_feature_vector]
        )
        return similarity_matrix.flatten()

    def get_similarity_of_feature_vector(
        self,
        feature_vector: NDArray[np.float64]
    ) -> float:
        similarity = self.__get_cosine_similarity_to_target([feature_vector])
        return similarity.tolist()[0]

    def get_similarity_of_feature_vectors(
        self,
        feature_vectors: Sequence[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        similarities = self.__get_cosine_similarity_to_target(feature_vectors)
        return similarities

    def __repr__(self):
        r_str = "CosineSimilarity()"
        return r_str


class RBFSimilarity(SOAPSimilarity):
    def __init__(
        self,
        target_feature_vector: NDArray[np.float64],
        rbf_gamma: float | None = None,
        adjust_gamma: bool = True,
        gamma_values: NDArray[np.float64] = np.logspace(-5, 1, 100),
        db_title: str = "RBFSimilarity"
    ):
        self.rbf_gamma = rbf_gamma
        self.target_feature_vector = target_feature_vector
        self.gamma_values = gamma_values
        self.adjust_gamma = adjust_gamma

        if self.rbf_gamma is None:
            self.rbf_gamma = 1 / len(self.target_feature_vector)

        if self.adjust_gamma is True:
            assert self.gamma_values is not None, \
                "If adjust_gamma is True, gamma_values must be given"

        self.n_calls = 0

        self.indiv_to_compare = []

        self.set_db_title(db_title)

    def __get_rbf_similarity_to_target(
        self,
        feature_vector_list: Sequence[NDArray[np.float64]],
    ) -> NDArray[np.float64]:

        similarity_matrix = rbf_kernel(
            feature_vector_list,
            [self.target_feature_vector],
            gamma=self.rbf_gamma
        )
        return similarity_matrix.flatten()

    def get_similarity_of_feature_vector(
        self,
        feature_vector: NDArray[np.float64]
    ) -> float:
        similarity = self.__get_rbf_similarity_to_target([feature_vector])
        return similarity.tolist()[0]

    def get_similarity_of_feature_vectors(
        self,
        feature_vectors: Sequence[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        similarities = self.__get_rbf_similarity_to_target(feature_vectors)
        return similarities

    def adjust_to_population(self, population: Sequence[NDArray[np.float64]]):
        if self.adjust_gamma:
            print(f"{self.get_db_title()}: Adjusting gamma to population")
            try:
                self.rbf_gamma = find_best_gamma_for_target_comparison(
                    population,
                    self.target_feature_vector,
                    self.gamma_values,
                )
            except Exception as e:
                warnings.warn(
                    "Gamma could not be adjusted. "
                    "This might cause the similarity to be suboptimal."
                    f"Error: {e}"
                )
        else:
            warnings.warn(
                "Gamma is not adjusted to the population. "
                "This might cause the similarity to be suboptimal."
                "Set adjust_gamma=True to adjust gamma to the population."
            )

    def set_db_title(self, title: str) -> None:
        super().set_db_title(title)

    def get_db_title(self) -> str:
        return self.db_title

    def __repr__(self):
        r_str = "RBFSimilarity("
        r_str += f"rbf_gamma={self.rbf_gamma})"
        return r_str


class AverageKernelSimilarity(SOAPSimilarity):
    def __init__(
        self,
        target_feature_vector: NDArray[np.float64],
        soap_object: CustomSOAP
    ):
        self.target_feature_vector = target_feature_vector
        self.soap_object = soap_object
        self.kernel = AverageKernel(soap_object)

    def __get_average_kernel_similarity_to_target(
        self,
        feature_vector_list: Sequence[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        similarity_matrix = self.kernel.create(
            feature_vector_list, [self.target_feature_vector])

        return similarity_matrix.flatten()

    def get_similarity_of_feature_vector(
        self,
        feature_vector: NDArray[np.float64]
    ) -> float:
        similarity = self.__get_average_kernel_similarity_to_target(
            [feature_vector]
        )
        return similarity.tolist()[0]

    def get_similarity_of_feature_vectors(
        self,
        feature_vectors: Sequence[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        similarities = self.__get_average_kernel_similarity_to_target(
            feature_vectors)
        return similarities

    def __repr__(self) -> str:
        r_str = "AverageKernelSimilarity()"
        return r_str


class NumberOfSameEntriesSimilarity(SOAPSimilarity):
    def __init__(
        self,
        target_feature_vector: NDArray[np.float64],
        relative_tolerance: float = 0.01,
        normalization: bool = False
    ) -> None:
        self.target_feature_vector = target_feature_vector
        self.relative_tolerance = relative_tolerance
        self.normalization = normalization

    def __get_number_of_same_entries_to_target(
        self,
        feature_vector_list: Sequence[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        similarities = np.sum(
            np.isclose(
                feature_vector_list,
                self.target_feature_vector,
                rtol=self.relative_tolerance
            ),
            axis=1
        )

        if self.normalization:
            similarities /= len(self.target_feature_vector)

        return similarities

    def get_similarity_of_feature_vector(
        self,
        feature_vector: NDArray[np.float64]
    ) -> float:
        similarity = self.__get_number_of_same_entries_to_target([
            feature_vector
        ])
        return similarity.tolist()[0]

    def get_similarity_of_feature_vectors(
        self,
        feature_vectors: Sequence[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        similarities = self.__get_number_of_same_entries_to_target(
            feature_vectors)
        return similarities

    def __repr__(self) -> str:
        r_str = "NumberOfSameEntriesSimilarity("
        r_str += f"relative_tolerance={self.relative_tolerance}, "
        r_str += f"normalization={self.normalization})"
        return r_str


class SpeciesSpecificRBFSim(SOAPSimilarity):
    def __init__(
        self,
        target_feature_vector: NDArray[np.float64],
        soap_object: CustomSOAP,
        species: str,
        species_to_compare: list[str],
        rbf_gamma: float | None = None,
        adjust_gamma: bool = True,
        gamma_values: NDArray[np.float64] = np.logspace(-1, 4, 100),
        db_title: str = "SpeciesSpecificRBFSim"
    ):
        self.rbf_gamma = rbf_gamma
        self.soap_object = soap_object
        self.target_feature_vector = target_feature_vector
        self.gamma_values = gamma_values
        self.adjust_gamma = adjust_gamma
        self.species = species
        self.species_to_compare = species_to_compare

        self.n_species_to_compare = len(species_to_compare)

        if self.rbf_gamma is None:
            self.rbf_gamma = 1 / len(self.target_feature_vector)

        if self.adjust_gamma is True:
            assert self.gamma_values is not None, \
                "If adjust_gamma is True, gamma_values must be given"

        db_title = db_title+"_"+species
        for species in species_to_compare:
            db_title += f"_{species}"
        self.set_db_title(db_title)

    def __repr__(self) -> str:
        r_str = "SpeciesSpecificRBFSim("
        r_str += f"rbf_gamma={self.rbf_gamma}, "
        r_str += f"species={self.species}, "
        r_str += f"species_to_compare={self.species_to_compare})"
        return r_str

    def set_db_title(self, title: str) -> None:
        super().set_db_title(title)

    def get_db_title(self) -> str:
        return self.db_title

    def __get_rbf_sim_for_species(
        self,
        feature_vector_list: Sequence[NDArray[np.float64]],
        species_to_compare: str
    ) -> NDArray[np.float64]:
        species_slice = self.soap_object.get_location(
            (self.species, species_to_compare)
        )

        species_reduced_feature_vectors = []
        for feature_vector in feature_vector_list:
            feature_vector_copy = feature_vector.copy()
            species_reduced_feature_vectors.append(
                feature_vector_copy[species_slice]
            )

        similarity_matrix = rbf_kernel(
            species_reduced_feature_vectors,
            [self.target_feature_vector[species_slice]],
            gamma=self.rbf_gamma
        )

        return similarity_matrix.flatten()

    def __get_rbf_sim_for_all_species(
        self,
        feature_vectors: Sequence[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        species_similarities = np.zeros(len(feature_vectors))
        for species in self.species_to_compare:
            species_similarity = self.__get_rbf_sim_for_species(
                feature_vectors,
                species
            )/self.n_species_to_compare

            species_similarities += species_similarity

        return species_similarities

    def get_similarity_of_feature_vector(
        self,
        feature_vector: NDArray[np.float64]
    ) -> float:
        similarity = self.__get_rbf_sim_for_all_species([feature_vector])
        return similarity.tolist()[0]

    def get_similarity_of_feature_vectors(
        self,
        feature_vectors: Sequence[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        similarities = self.__get_rbf_sim_for_all_species(feature_vectors)
        return similarities

    def adjust_to_population(self, population: Sequence[NDArray[np.float64]]):
        if self.adjust_gamma:
            print(f"{self.get_db_title()}: Adjusting gamma to population")
            self.rbf_gamma = find_best_gamma_for_target_comparison(
                population,
                self.target_feature_vector,
                self.gamma_values,
            )
        else:
            warnings.warn(
                "Gamma is not adjusted to the population. "
                "This might cause the similarity to be suboptimal."
                "Set adjust_gamma=True to adjust gamma to the population."
            )
