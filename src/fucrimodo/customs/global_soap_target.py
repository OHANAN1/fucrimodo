from typing import Literal, overload
import warnings

import ase
import numpy as np
from dscribe.descriptors import SOAP
from ase.data import chemical_symbols
from fucrimodo.core import Individual
from numpy.typing import NDArray


class GlobalSOAP:
    """Wrapper for :class:`dscribe.descriptors.SOAP` that produces global SOAP descriptors.

    This wrapper adds quality-of-life features over the base SOAP class:

    * ``species`` accepts both chemical symbols and atomic numbers.
    * :meth:`get_init_params` returns the parameters used to instantiate the class.
    * :meth:`create` validates the input structure before computing the descriptor.
      and is stricter with the return shape.
    * :meth:`get_present_species` return the species that truely contribute to a given
      SOAP feature vector.


    For more info see the `DScribe documentation <https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html>`__.

    :param r_cut: Cutoff radius for the local environment.
    :param n_max: Number of radial basis functions.
    :param l_max: Maximum degree of spherical harmonics.
    :param species: List of chemical symbols or atomic numbers to include.
    :param average: Averaging mode for the global descriptor. Must be ``"inner"`` or ``"outer"``.
    :param sigma: Width of the Gaussian basis functions.
    :param periodic: Whether the structures are treated as periodic.

    :raises AssertionError: If ``average`` is ``"off"``.
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
            species=self._species,
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
        """Return the parameters used to initialise this wrapper.

        :return: Dictionary of initialisation parameters that can be passed to
            :class:`GlobalSOAP` to recreate the object.
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
        """Return the number of features in the SOAP descriptor."""
        return self._dscribe_soap.get_number_of_features()

    def get_location(self, species: tuple) -> slice:
        """Return the slice locating the features for the given species pair.

        :param species: Tuple of species defining the requested feature block.

        :return: Slice object indexing the feature block in the descriptor.
        """
        return self._dscribe_soap.get_location(species)

    def get_present_species(
        self,
        feature_vector: np.ndarray,
        sort_by_appearance: bool = True,
    ) -> list[str]:
        """Return the species that contribute to the given SOAP feature vector.

        A species is considered present if the sum of absolute values in its
        corresponding feature slice is non-zero.

        :param feature_vector: SOAP descriptor to analyse.
        :param sort_by_appearance: If ``True``, sort the returned species by their
            total feature contribution in descending order. This is not equal to
            their contribution in their original structure, but it can be used
            as an estimate.

        :return: List of species that have non-zero features.
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

    # Overloads for better type hinting
    @overload
    def create(
        self,
        system: Individual,
        n_jobs: int = 1,
        only_physical_cores: bool = False,
        verbose: bool = False,
    ) -> NDArray[np.float64]: ...

    # Overloads for better type hinting
    @overload
    def create(
        self,
        system: list[Individual],
        n_jobs: int = 1,
        only_physical_cores: bool = False,
        verbose: bool = False,
    ) -> list[NDArray[np.float64]]: ...

    def create(
        self,
        system: list[Individual] | Individual,
        n_jobs=1,
        only_physical_cores=False,
        verbose=False,
    ) -> list[NDArray[np.float64]] | NDArray[np.float64]:
        """Create the global SOAP descriptor for one or more structures.

        Validates the input before delegating to the underlying SOAP descriptor.

        :param system: Single structure or list of structures to describe.
        :param n_jobs: Number of parallel jobs.
        :param only_physical_cores: Use only physical CPU cores.
        :param verbose: Print verbose output.

        :return: SOAP descriptor array, or a list of arrays if multiple structures were provided.

        :raises ValueError: If any input structure is invalid or SOAP creation fails.
        """

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
            if type(system) is Individual or type(system) is ase.Atoms:
                return results  # type: ignore
            elif type(system) is list and len(system) == 1:
                return [results]  # type: ignore
            return results  # type: ignore
        except Exception as e:
            raise ValueError(f"Error in creating SOAP: {e}")
