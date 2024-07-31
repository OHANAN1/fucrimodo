from ase.ga.utilities import closest_distances_generator
from ase import data as ase_data
from ase.visualize import view
import numpy as np
import ase
from ase.ga import utilities as ase_utilities


def parallelepiped_heights(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    c: np.ndarray | list[float], 
    volume: float | None = None
) -> list[float]:

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    if volume is None:
        volume = np.abs(np.dot(a, np.cross(b, c)))
    
    areas = (
        np.linalg.norm(np.cross(b, c)).tolist(),
        np.linalg.norm(np.cross(a, c)).tolist(),
        np.linalg.norm(np.cross(a, b)).tolist()
    )

    heights: list[float] = []
    for i in range(3):
        if volume is not None:
            heights.append(volume / areas[i])

    return heights


class CustomClosestDistances(dict[tuple[int, int], float]):
    """
    Works like ase.ga.utilities.closest_distances_generator, but with custom
    functionality:
    - has better __repr__ method to print the bounds
    - Can be used as easier type hint for type checking
    - Can use chemical symbols as strings or atomic numbers as integers
      (dict is still made with atomic numbers)
      - assigns self._chemical_symbols to the chemical symbols
        and self._atomic_numbers to the atomic numbers
    - Can check if atoms are too close to each other in the unit cell
    - Also check if an atom is too close to itself in the neighboring unit cell
    - Also saves the covalent radii of the species in the species list
    """

    def __init__(
        self,
        species: list[int] | list[str],
        ratio_of_covalent_radii: float
    ):
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
                ase_data.chemical_symbols[specie]
                for specie in unique_species
            ]
        else:
            raise ValueError(
                "species must be either all integers or all strings"
            )

        self.ratio_of_covalent_radii = ratio_of_covalent_radii

        self._ase_closest_distances = closest_distances_generator(
            atom_numbers=self._atomic_numbers,
            ratio_of_covalent_radii=ratio_of_covalent_radii
        )

        self.covalent_radii_dict = self.get_colvalent_radii(
            self._atomic_numbers
        )

    def __call__(self) -> dict[tuple[int, int], float]:
        return self._ase_closest_distances

    def __getitem__(self, key: tuple[int, int]) -> float:
        return self._ase_closest_distances[key]

    def __repr__(self) -> str:
        r_str = "CustomClosestDistances("
        r_str += f"chemical_symbols={self._chemical_symbols}, "
        r_str += f"ratio_of_covalent_radii={self.ratio_of_covalent_radii}"
        r_str += ")"
        return r_str

    def __convert_species(self, species: list[int] | list[str]) -> list[int]:
        if all(isinstance(specie, str) for specie in species):
            return [ase_data.atomic_numbers[specie] for specie in species]
        elif all(isinstance(specie, int) for specie in species):
            return species  # type: ignore
        else:
            raise ValueError(
                "species must be either all integers or all strings"
            )

    def get_colvalent_radii(
        self, species: list[int] | list[str]
    ) -> dict[int, float]:
        """
        Returns the covalent radii of the species in the species list.
        """
        unique_species = np.unique(species)
        if isinstance(unique_species, np.ndarray):
            unique_species = unique_species.tolist()

        atomic_numbers = self.__convert_species(unique_species)

        covalent_radii = {}
        for atomic_number in atomic_numbers:
            if hasattr(ase_data, "covalent_radii_dict"):
                if atomic_number in self.covalent_radii_dict:
                    covalent_radii[atomic_number] = self.covalent_radii_dict[
                        atomic_number
                    ]
                else:
                    covalent_radii[atomic_number] = ase_data.covalent_radii[
                        atomic_number
                    ]

            else:
                covalent_radii[atomic_number] = ase_data.covalent_radii[
                    atomic_number
                ]

        return covalent_radii

    def atom_is_too_close_to_itself(self, crystal: ase.Atoms) -> bool:
        """
        Checks if the hight of the unit cell is smaller than the covalent 
        radii of the atoms in the crystal.
        If this is the case, the atoms are most often too close to themselves 
        in the neighboring unit cells.
        """
        atomic_numbers = np.unique(
            crystal.get_atomic_numbers().astype(int).tolist()
        )
        cell_vectors = np.array(crystal.get_cell()[:])  # type: ignore

        heights = parallelepiped_heights(
            cell_vectors[0],
            cell_vectors[1],
            cell_vectors[2]
        )

        min_height = min(heights)
        for number in atomic_numbers:
            closest_distance = self.covalent_radii_dict[number]
            if min_height < closest_distance:
                return True

        return False

    def atoms_are_too_close(self, crystal: ase.Atoms) -> bool:
        """
        Uses the atoms too close function from ASE, but also checks if the atoms
        are too close to themselves in the neighboring unit cells.
        """
        if ase_utilities.atoms_too_close(crystal, self._ase_closest_distances):
            return True

        if self.atom_is_too_close_to_itself(crystal):
            return True

        return False

if __name__ == "__main__":

    crystal = ase.Atoms(
        numbers=[13, 14],
        positions=[[ 2.77454447, 2.00247484, 1.15038187], [-3.77171666,  5.00409698,  2.89335887]],
        cell=[[5.045397301687402, 0.0, 0.0], [0.6189786398140509, 3.0052403155179395, 1.6455890243462532], [-8.225867019276428, 4.423720153210335, 2.614187448699669]],
        pbc=True
    )
    crystal.wrap()

    cell_vectors = crystal.get_cell()[:]  # type: ignore
    print("Cell vectors:", cell_vectors)
    heights = parallelepiped_heights(
        cell_vectors[0],
        cell_vectors[1],
        cell_vectors[2]
    )
    print("Heights:", heights)
    print("Min height:", min(heights) * 2)
    closest_distances = CustomClosestDistances(
        species=[13, 14],
        ratio_of_covalent_radii=0.9
    )
    print(closest_distances._ase_closest_distances)
    print("Covalent radii:", closest_distances.covalent_radii_dict)
    print("Ase utility:", ase_utilities.atoms_too_close(
        crystal, closest_distances._ase_closest_distances
    ))
    print("To itself:", closest_distances.atom_is_too_close_to_itself(crystal))
    print(closest_distances.atoms_are_too_close(crystal))
    view(crystal)

