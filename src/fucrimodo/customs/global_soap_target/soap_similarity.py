import warnings
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from fucrimodo.core import Individual
from numpy.typing import NDArray
from sklearn.metrics.pairwise import rbf_kernel

from .global_soap import GlobalSOAP


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
        self, feature_vectors: Sequence[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def set_db_title(self, title: str) -> None:
        """
        Optional function that can be used to set the title of the
        ase database. Therefore no space, no numbers and no special
        characters are allowed. '_' is allowed.
        """
        forbidden_chars = [" ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
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
# ║                   Similarity Classes                     ║
# ╚══════════════════════════════════════════════════════════╝


class RBFSimilarity(SOAPSimilarity):
    def __init__(
        self,
        target_feature_vector: NDArray[np.float64],
        descriptor_object: GlobalSOAP | None = None,
        rbf_gamma: float = 0.1,
        db_title: str = "RBFSimilarity",
    ):
        self.rbf_gamma = rbf_gamma
        self.target_feature_vector = target_feature_vector
        self.descriptor_object = descriptor_object

        self.set_db_title(db_title)

    def __get_rbf_similarity_to_target(
        self,
        feature_vector_list: Sequence[NDArray[np.float64]],
    ) -> NDArray[np.float64]:

        similarity_matrix = rbf_kernel(
            feature_vector_list, [self.target_feature_vector], gamma=self.rbf_gamma
        )
        return similarity_matrix.flatten()

    def get_similarity_of_feature_vector(
        self, feature_vector: NDArray[np.float64]
    ) -> float:
        similarity = self.__get_rbf_similarity_to_target([feature_vector])
        return similarity.tolist()[0]

    def get_similarity_of_feature_vectors(
        self, feature_vectors: Sequence[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        similarities = self.__get_rbf_similarity_to_target(feature_vectors)
        return similarities

    def set_db_title(self, title: str) -> None:
        super().set_db_title(title)

    def get_db_title(self) -> str:
        return self.db_title

    def __repr__(self):
        r_str = "RBFSimilarity("
        r_str += f"rbf_gamma={self.rbf_gamma})"
        return r_str


class SpeciesSpecificRBFSim(SOAPSimilarity):
    """Automatically sets the db title"""

    def __init__(
        self,
        target_feature_vector: NDArray[np.float64],
        soap_object: GlobalSOAP,
        species: tuple[str, str | list[str]],
        rbf_gamma: float | None = None,
        adjust_gamma: bool = False,
        gamma_values: NDArray[np.float64] = np.logspace(-1, 4, 100),
        db_title: None = None,
    ):
        super()

        self.rbf_gamma = rbf_gamma
        self.soap_object = soap_object
        self.target_feature_vector = target_feature_vector
        self.gamma_values = gamma_values
        self.adjust_gamma = adjust_gamma
        self.species = species

        self.n_species_to_compare = len(species_to_compare)

        if self.rbf_gamma is None:
            self.rbf_gamma = 1 / len(self.target_feature_vector)

        if self.adjust_gamma is True:
            assert (
                self.gamma_values is not None
            ), "If adjust_gamma is True, gamma_values must be given"

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
        species_to_compare: str,
    ) -> NDArray[np.float64]:
        species_slice = self.soap_object.get_location(
            (self.species, species_to_compare)
        )

        species_reduced_feature_vectors = []
        for feature_vector in feature_vector_list:
            feature_vector_copy = feature_vector.copy()
            species_reduced_feature_vectors.append(feature_vector_copy[species_slice])

        similarity_matrix = rbf_kernel(
            species_reduced_feature_vectors,
            [self.target_feature_vector[species_slice]],
            gamma=self.rbf_gamma,
        )

        return similarity_matrix.flatten()

    def __get_rbf_sim_for_all_species(
        self, feature_vectors: Sequence[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        species_similarities = np.zeros(len(feature_vectors))
        for species in self.species_to_compare:
            species_similarity = (
                self.__get_rbf_sim_for_species(feature_vectors, species)
                / self.n_species_to_compare
            )

            species_similarities += species_similarity

        return species_similarities

    def get_similarity_of_feature_vector(
        self, feature_vector: NDArray[np.float64]
    ) -> float:
        similarity = self.__get_rbf_sim_for_all_species([feature_vector])
        return similarity.tolist()[0]

    def get_similarity_of_feature_vectors(
        self, feature_vectors: Sequence[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        similarities = self.__get_rbf_sim_for_all_species(feature_vectors)
        return similarities
