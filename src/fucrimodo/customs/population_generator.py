import random
import ase
from fucrimodo.core.modules import PopulationGenerator, Individual
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from ase.ga.startgenerator import StartGenerator
import warnings
import ase
import numpy as np

import logging
logger = logging.getLogger('run_logger')


def convert_ase_atoms_to_individual(atoms: ase.Atoms) -> Individual:
    return Individual(
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        pbc=atoms.pbc,
        symbols=atoms.get_chemical_symbols(),
    )

class OneAtomicCrystalGenerator(PopulationGenerator):
    """Class to generate a population of one atomic crystals.

    :param atom_types: A list of atom types that are used to generate the
    :param cell_bounds: The bounds of the cell parameters which the
        generated crystals should not exceed or be below of.
    :param closest_distances: The closest distances that define the
        minimum allowed distance between atoms.
    :param volume: The volume of the generated crystals.
    """

    def __init__(
        self,
        atom_types: list[str],
        cell_bounds: CustomCellBounds,
        closest_distances: CustomClosestDistances,
        volume: float,
    ):
        self.atom_types = atom_types
        self.cell_bounds = cell_bounds
        self.closest_distances = closest_distances
        self.volume = volume

    def __generate_individuals_with_specific_species(
        self,
        n: int,
        species: list[str],
    ) -> list[Individual]:

        a_min_max = self.cell_bounds.bounds["a"]
        b_min_max = self.cell_bounds.bounds["b"]
        c_min_max = self.cell_bounds.bounds["c"]

        step = 0
        max_steps = 2 * n
        inds = []
        while len(inds) < n:
            cell_vectors = [
                [np.random.uniform(a_min_max[0], a_min_max[1]), 0, 0],
                [0, np.random.uniform(b_min_max[0], b_min_max[1]), 0],
                [0, 0, np.random.uniform(c_min_max[0], c_min_max[1])]
            ]
            ase_atoms = ase.Atoms(
                symbols=species,
                positions=[[0, 0, 0]],
                cell=cell_vectors,
                pbc=True
            )

            if not self.closest_distances.atoms_are_too_close(ase_atoms):
                inds.append(
                    convert_ase_atoms_to_individual(ase_atoms)
                )

            step += 1

            if step > max_steps:
                warnings.warn(
                    "Could not generate {} individuals".format(n), UserWarning
                )
                break

        return inds

    def generate_individuals(self, n: int) -> list[Individual]:
        # Generate individuals
        individuals = []
        for atom_type in self.atom_types:
            inds = self.__generate_individuals_with_specific_species(
                n // len(self.atom_types),
                [atom_type]
            )
            individuals.extend(inds)

        # If not enough individuals were generated, generate the rest with
        # random species
        step = 0
        max_steps = 2 * n
        while len(individuals) < n:
            atom_type = random.choice(self.atom_types)
            inds = self.__generate_individuals_with_specific_species(
                1,
                [atom_type]
            )
            individuals.extend(inds)

            step += 1
            if step > max_steps:
                warnings.warn(
                    "Could not generate {} individuals".format(n), UserWarning
                )
                break

        return individuals
