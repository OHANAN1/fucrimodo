import random
import ase
from fucrimodo.core.modules import PopulationGenerator, Individual
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from ase.ga.startgenerator import StartGenerator
import warnings
import ase

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

    def generate_individuals(self, n: int) -> list[Individual]:
        # Create a generator for each atom type
        # This is just how the ase.ga.startgenerator.StartGenerator works
        # block defines the number of atoms of each type, here 1
        generators = [StartGenerator(
            slab=ase.Atoms('', pbc=True),
            blocks=[(atom_type, 1)],
            blmin=self.closest_distances._ase_closest_distances,
            number_of_variable_cell_vectors=3,
            cellbounds=self.cell_bounds._ase_cellbounds,
            box_volume=self.volume,
            splits={(2,): 1, (1,): 1},
            test_dist_to_slab=False,
            test_too_far=False,
        ) for atom_type in self.atom_types]

        max_steps = 2 * n
        # Generate individuals
        step = 0
        individuals = []
        while len(individuals) < n:
            # Get a random generator for a specific atom type
            gen_index = random.randint(0, len(generators) - 1)

            # Get a new candidate from the generator
            crystal = generators[gen_index].get_new_candidate(maxiter=1000)

            # The generator returns None, if the crystal could not be created
            # in the internal max number of steps
            if crystal is not None:

                # Convert the ase.Atoms object to an Individual object
                individuals.append(
                    convert_ase_atoms_to_individual(crystal)
                )

            # Increase step and check if max steps is reached
            step += 1
            if step > max_steps:
                warnings.warn(
                    "Could not generate {} individuals".format(n), UserWarning
                )
                break

        return individuals
