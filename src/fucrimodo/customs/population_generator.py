import logging
import random
import warnings
from collections.abc import Iterable
from logging import Logger
from multiprocessing import Pool
from typing import Sequence

import ase
import numpy as np
from ase.build import sort
from fucrimodo.core import utils as core_utils
from fucrimodo.core.modules import Individual, Population, PopulationGenerator
from fucrimodo.core.modules.fitness_function import FitnessFunction
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.utils.custom_soap import CustomSOAP
from fucrimodo.customs import global_soap_target
from fucrimodo.customs import population_generator as pop_gen
from fucrimodo.customs import population_selections
from pyxtal import pyxtal
from pyxtal.symmetry import Group
from pyxtal.tolerance import Tol_matrix

logger = logging.getLogger("run_logger")


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
                [0, 0, np.random.uniform(c_min_max[0], c_min_max[1])],
            ]
            ase_atoms = ase.Atoms(
                symbols=species, positions=[[0, 0, 0]], cell=cell_vectors, pbc=True
            )

            if not self.closest_distances.atoms_are_too_close(ase_atoms):
                inds.append(convert_ase_atoms_to_individual(ase_atoms))

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
                n // len(self.atom_types), [atom_type]
            )
            individuals.extend(inds)

        # If not enough individuals were generated, generate the rest with
        # random species
        step = 0
        max_steps = 2 * n
        while len(individuals) < n:
            atom_type = random.choice(self.atom_types)
            inds = self.__generate_individuals_with_specific_species(1, [atom_type])
            individuals.extend(inds)

            step += 1
            if step > max_steps:
                warnings.warn(
                    "Could not generate {} individuals".format(n), UserWarning
                )
                break

        return individuals


# TODO: Make this part of the RandomSampleCrystalPopulation
# Either as static or private method
def create_random_structure(
    present_species: list[str],
    n_atoms: int,
    closest_distances: CustomClosestDistances,
    possible_space_groups: Iterable[int] = range(1, 231),
    logger: Logger | None = None,
    n_retries: int = 1000,
    seed: int = 42,
    sort_composition_descending: bool = True,
) -> ase.Atoms | None:
    """Worker function to create a random structure."""
    # Set the random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    # Preset values
    atoms = None
    space_group = 1
    n_present_species = len(present_species)
    n_atoms_per_species = [1 for _ in range(0, n_present_species)]

    if logger is not None:
        logger.debug(f"Creating random crystal with seed={seed}.")

    # Try different combinations of compositions and space groups until
    # a valid one is found
    while n_retries > 0:

        # Ensure that every species is present in the crystal at least once
        n_atoms_per_species = [1 for _ in range(0, n_present_species)]

        # Distribute the remaining atoms randomly
        remaining = n_atoms - n_present_species
        for _ in range(0, remaining):
            index_to_add = rng.integers(0, n_present_species)
            n_atoms_per_species[index_to_add] += 1

            space_group = random.choice(list(possible_space_groups))

        if sort_composition_descending:
            # Sort the composition in descending order.
            # This can be useful if the present species are sorted in
            # descending order of their guessed appearance in the
            # target features.
            n_atoms_per_species = sorted(n_atoms_per_species, reverse=True)

        # Check if the space group is compatible with the number of atoms
        compatible, _ = Group(space_group, dim=3).check_compatible(n_atoms_per_species)
        if not compatible:
            n_retries -= 1
        else:
            break

    try:
        tol_matrix = Tol_matrix(prototype="atomic", factor=1.0)
        # Create random crystal
        xtal = pyxtal(random_state=rng)
        xtal.from_random(
            3,
            space_group,
            present_species,
            n_atoms_per_species,
            seed=seed,
            random_state=rng,
            max_count=10,
            force_pass=False,
            conventional=False,
            tm=tol_matrix,
        )
        atoms = xtal.to_ase()
        assert (
            type(atoms) is ase.Atoms
        ), f"Generated object is not of type ase.Atoms, but {type(atoms)}."

        atoms.set_pbc([True, True, True])
        if closest_distances.atoms_are_too_close(atoms):
            if logger is not None:
                logger.info(
                    "Could not create structure with correct distances.",
                )
            atoms = None

    except RuntimeError:
        # If the crystal could not be created, because it took too long
        # just return None
        if logger is not None:
            logger.error("Could not create random crystal in an appropriate time.")
            logger.debug(f"Space group: {space_group}")
            logger.debug(f"Number of atoms per species: {n_atoms_per_species}")
            logger.debug(f"Present species: {present_species}")
        else:
            print("Could not create random crystal in an appropriate time.")
            print(f"Space group: {space_group}")
            print(f"Number of atoms per species: {n_atoms_per_species}")
            print(f"Present species: {present_species}")
        atoms = None

    return atoms


class RandomSampleCrystalPopulation(PopulationGenerator):
    def __init__(
        self,
        soap_obj: CustomSOAP,
        target_features: np.ndarray,
        closest_distances: CustomClosestDistances,
        n_atoms,
        fitness_functions: Sequence[FitnessFunction | tuple[FitnessFunction, float]],
        n_samples: int = 1000,
        n_jobs: int = 1,
        exclude_space_groups: Iterable[int] | None = [
            215,
            195,
        ],  # Exclude 215 and 195, since they somehow cause problems
        rng=np.random.default_rng(),
        logger: None | Logger = None,
        ensure_unique_individuals: bool = True,
    ):
        self.soap_obj = soap_obj
        self.target_features = target_features
        self.closest_distances = closest_distances
        self.fitness_functions = fitness_functions
        self.n_samples = n_samples
        self.n_jobs = n_jobs
        self.ensure_unique_individuals = ensure_unique_individuals
        self.exclude_space_groups = exclude_space_groups
        self.n_atoms = n_atoms
        self.rng = rng
        self.logger = logger

        self.possible_space_groups = self._get_possible_space_groups(n_atoms)

        # Get species present in soap, sorted by their estimated apperance
        self.present_species = global_soap_target.utils.get_present_species(
            soap_obj=soap_obj,
            feature_vector=target_features,
            sort_by_appearance=True,
        )

    def _get_possible_space_groups(self, n_atoms) -> list:
        # Use every space group but the ones that are not compatible with the
        # current number of atoms
        space_groups = range(1, 231)

        if self.exclude_space_groups:
            space_groups = [
                sg for sg in space_groups if sg not in self.exclude_space_groups
            ]
            if self.logger:
                self.logger.debug(
                    f"Excluding space groups: {self.exclude_space_groups}"
                )

        space_groups = [
            sg
            for sg in space_groups
            if Group(sg, dim=3).check_compatible([1 for _ in range(0, n_atoms)])[0]
        ]

        return space_groups

    def generate_individuals(self, n: int) -> list[Individual]:
        if self.logger:
            self.logger.info(f"Creating random crystals with n_atoms={self.n_atoms}...")

        results = []
        if self.n_jobs > 1:
            with Pool(self.n_jobs) as pool:
                results = pool.starmap(
                    create_random_structure,
                    [
                        (
                            self.present_species,
                            self.n_atoms,
                            self.closest_distances,
                            self.possible_space_groups,
                            None,
                            1000,
                            i,
                            True,
                        )
                        for i in range(1, self.n_samples)
                    ],
                )
        else:
            for i in range(1, self.n_samples):
                results = [
                    create_random_structure(
                        self.present_species,
                        self.n_atoms,
                        self.closest_distances,
                        self.possible_space_groups,
                        None,
                        1000,
                        i,
                        True,
                    )
                    for i in range(1, self.n_samples)
                ]

        if self.logger is not None:
            self.logger.info(f"{self.n_atoms}: Finished creating random crystals.")

        # Sort out None values, for which no crystal could be created
        random_crystals: list[ase.Atoms] = []
        for i, result in enumerate(results):
            if result is None:
                if self.logger is not None:
                    self.logger.debug(
                        f"{self.n_atoms}: " f"Could not create random crystal {i + 1}."
                    )
            else:
                random_crystals.append(result)

        # Convert the crystals to individuals
        individuals = []
        for crystal in random_crystals:
            ind = pop_gen.convert_ase_atoms_to_individual(crystal)
            individuals.append(ind)

        if self.logger:
            self.logger.info(
                f"Generated {len(individuals)} individuals for n_atoms={self.n_atoms}"
            )

        if self.ensure_unique_individuals:
            unique_inds = []
            for ind in individuals:
                ind = sort(ind)
                if ind not in unique_inds:
                    unique_inds.append(ind)

            individuals = unique_inds

            if self.logger:
                self.logger.info(f"{len(individuals)} unique individuals found.")

        return individuals

    def generate_population(self, size: int) -> Population:
        pop = Population(individuals=self.generate_individuals(size))

        if self.logger:
            self.logger.info(f"Selecting best {size} individuals.")

        core_utils.population_utils.assign_fitness_to_all_individuals(
            population=pop,
            fitness_functions=self.fitness_functions,
        )

        pop.individuals = population_selections.NSGA2Selection().select(
            pop.individuals, size
        )

        # Copy random inds if not enough ind could be generated
        if len(pop.individuals) < size:
            if self.logger:
                self.logger.warning(
                    f"Could only generate {len(pop.individuals)}/{size} "
                    "individuals. Extending population."
                )
            extended_inds = pop.individuals
            while len(extended_inds) <= size:
                choosen_ind = random.choice(extended_inds).copy()
                extended_inds.append(choosen_ind)

        return pop
