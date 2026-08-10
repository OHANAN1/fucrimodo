from typing import Literal
import warnings

import ase
import numpy as np
from dscribe.descriptors import SOAP
from ase.data import chemical_symbols
from fucrimodo.core import Individual


class GlobalSOAP:
    """Wrapper for `dscribe.descriptors.SOAP`.

    This wrapper is implemented to make it clearer, what the object returns. Additionally adds some QoL features:
    1. Species attribute accept chemical symbols and numbers.
    2. Call GlobalSOAP.get_init_params() to get dict of params used for __init__.
    3. On GlobaSOAP.create(...) automatically checks if atoms object is valid and can be calculated properly.
    """

    def __init__(
        self,
        r_cut: float,
        n_max: int,
        l_max: int,
        species: list[str] | list[int],
        average: Literal["inner", "outer"] = "inner",
        sigma: float = 1.0,
        periodic: bool = True,
    ) -> None:

        # Ensure that averaging is set, else its not a global soap
        assert (
            average != "off"
        ), "Cannot turn averaging off for the global SOAP, please use values 'inner' or 'outer'."

        self.r_cut = r_cut
        self.n_max = n_max
        self.l_max = l_max
        self.sigma = sigma
        self.periodic = periodic
        self.average = average

        # Convert the int species to string species
        str_species: list[str] = []
        for s in species:
            if isinstance(s, int):
                str_species.append(chemical_symbols[s])
            elif isinstance(s, str):
                str_species.append(s)
            else:
                raise ValueError("Species must be a list of strings or integers")

        self._species = str_species

        self._dscribe_soap = SOAP(
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            species=species,
            sigma=sigma,
            periodic=periodic,
            average=average,
            sparse=False,
        )

    @property
    def species(self) -> list[str]:
        """List of chemical symbols used to calculate the SOAP descriptor."""
        return self._species

    def get_init_params(self) -> dict:
        """
        Returns a dictionary with parameters that where used to set up the
        class.

        Example:

        soap_params = global_soap.get_init_params()
        global_soap_copy = GlobalSOAP(**soap_params)
        """
        return {
            "r_cut": self.r_cut,
            "n_max": self.n_max,
            "l_max": self.l_max,
            "species": self.species,
            "sigma": self.sigma,
            "periodic": self.periodic,
            "average": self.average,
        }

    def __is_valid(self, structure: ase.Atoms) -> bool:
        if not isinstance(structure, ase.Atoms):
            warnings.warn("Input is not an ASE Atoms object")
            return False

        atomic_numbers = structure.get_atomic_numbers()
        if np.isnan(atomic_numbers).any():
            warnings.warn("Atomic numbers contain NaNs")
            return False

        positions = structure.get_positions()
        if np.isnan(positions).any():
            warnings.warn("Positions contain NaNs")
            return False

        if len(atomic_numbers) != len(positions):
            warnings.warn("Number of atomic numbers and positions do not match")
            return False

        if len(atomic_numbers) == 0:
            warnings.warn("No atoms in the structure")
            return False

        if not hasattr(structure, "cell"):
            warnings.warn("Cell is not defined")
            return False

        cell = structure.get_cell()[:]  # type: ignore
        if np.nan in cell:
            warnings.warn("Cell contains NaNs")
            return False

        return True

    def get_number_of_features(self) -> int:
        return self._dscribe_soap.get_number_of_features()

    def get_location(self, species: tuple) -> slice:
        return self._dscribe_soap.get_location(species)

    def create(
        self,
        system: list[Individual] | Individual,
        n_jobs=1,
        only_physical_cores=False,
        verbose=False,
    ) -> list[np.ndarray]:
        """Always returns list of features"""
        if isinstance(system, Individual):
            if not self.__is_valid(system):
                raise ValueError("Invalid input. Check warnings for details.")
        else:
            if not all(self.__is_valid(structure) for structure in system):
                raise ValueError("Invalid input. Check warnings for details.")

        try:
            results = self._dscribe_soap.create(
                system=system,
                n_jobs=n_jobs,
                only_physical_cores=only_physical_cores,
                verbose=verbose,
            )
            if type(system) is not list:
                return [results]  # type: ignore
            elif type(system) is list and len(system) == 1:
                return [results]  # type: ignore
            return results  # type: ignore
        except Exception as e:
            raise ValueError(f"Error in creating SOAP: {e}")

    def get_present_species(
        self,
        feature_vector: np.ndarray,
        sort_by_appearance: bool = True,
    ) -> list[str]:
        """Return only those species that contribute to the soap.

        This means all species that have values != 0 in their corresponding
        slice of the whole feature vector.

        :param soap_obj: The soap object that was used to create the
            target features.
        :param target_features: The target soap features.
        :param sort_by_appearance: If set to True, the method tries to guess
            the approximate composition of the target structure by calculating the
            total number of features for each species. The output is then sorted
            by the number of features in descending order.

        :return: A list of species that have features in the target soap.
            If sort_by_appearance is set to True, the list is sorted by the
            number of features in descending order. (The most prominent species
            is at index 0.)
        """
        # Ensure that the analysis is only done once for each species
        unique_soap_obj_species = list(set(self.species))

        # Loop over all unique species in the soap object
        species_with_features: list[str] = []
        feature_sum_per_species: list[float] = []
        for single_specie in unique_soap_obj_species:
            # Get the slice obj for the part of the feature vector that
            # corresponds to the current species
            species_slice = self.get_location((single_specie, single_specie))

            # Calculate the sum of absolute values of the feature vector
            feature_vec_abs_sum = float(np.sum(np.abs(feature_vector[species_slice])))

            # If the sum is zero, the species has no features in the provided
            # feature vector
            if feature_vec_abs_sum == 0:
                print(f"Species {single_specie} has no features in descriptor.")
                # continue to the next species so it is not added to the list
                # of species with features
                continue

            else:
                # Add the species to the list of species with features
                species_with_features.append(single_specie)
                feature_sum_per_species.append(feature_vec_abs_sum)

        # If the sort_by_appearance flag is set to True, sort the species
        # by the number of features
        if sort_by_appearance:
            sort_indices = np.argsort(feature_sum_per_species)

            # Reverse the sort order, so that it is from high to low
            sort_indices = sort_indices[::-1]
            species_with_features = [species_with_features[i] for i in sort_indices]

        return species_with_features
