from fucrimodo.core.modules.individual import Individual
import numpy as np
from ase import data as ase_data
from ase_ga import utilities as ase_utilities


class CustomClosestDistances(dict[tuple[int, int], float]):
    """Check if atoms are too close to each other based on coval radii.

    Class that works like ase_ga.utilities.closest_distances_generator, but with custom
    functionality:

    * has __repr__ method to print the bounds
    * Can use chemical symbols as strings or atomic numbers as integers
      (dict is still made with atomic numbers)
    * Method 'atoms_are_too_close' checks if atoms in an individual are too close to each other
      (Also check if atoms are too close to themselves for periodic boundary conditions)
    * Can be used as easier type hint for type checking

    """

    def __init__(self, species: list[int] | list[str], ratio_of_covalent_radii: float):
        unique_species = np.unique(species)
        if isinstance(unique_species, np.ndarray):
            unique_species = unique_species.tolist()

        if all(isinstance(specie, str) for specie in unique_species):
            self._atomic_numbers = [
                ase_data.atomic_numbers[specie] for specie in unique_species
            ]
            self._chemical_symbols = unique_species
        elif all(isinstance(specie, int) for specie in unique_species):
            self._atomic_numbers = unique_species
            self._chemical_symbols = [
                ase_data.chemical_symbols[specie] for specie in unique_species
            ]
        else:
            raise ValueError("species must be either all integers or all strings")

        self.ratio_of_covalent_radii = ratio_of_covalent_radii

        self._ase_closest_distances = ase_utilities.closest_distances_generator(
            atom_numbers=self._atomic_numbers,
            ratio_of_covalent_radii=ratio_of_covalent_radii,
        )

        # Populate the dict class
        super().__init__(self._ase_closest_distances)

    def __repr__(self) -> str:
        r_str = "CustomClosestDistances("
        r_str += f"chemical_symbols={self._chemical_symbols}, "
        r_str += f"ratio_of_covalent_radii={self.ratio_of_covalent_radii}"
        r_str += ")"
        return r_str

    def atoms_are_too_close(self, individual: Individual) -> bool:
        """Checks if the atoms in are structure are too close.

        Uses the atoms too close function from ASE, which also checks if the atoms
        are too close to themselves in the neighboring unit cells if pbc enabled.
        """
        return ase_utilities.atoms_too_close(individual, self._ase_closest_distances)
