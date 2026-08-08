import logging
import random
from collections.abc import Iterable
from logging import Logger
from multiprocessing import Pool
from typing import Sequence

import ase
import numpy as np
from ..core import utils as core_utils
from ..core import Individual
from ..core.abstracts import PopulationGenerator, FitnessFunction
from ..core.utils import CustomClosestDistances
from .global_soap_target import GlobalSOAP
from . import population_selections
from . import global_soap_target
from .utils import convert_ase_atoms_to_individual
from pyxtal import pyxtal
from pyxtal.symmetry import Group
from pyxtal.tolerance import Tol_matrix

logger = logging.getLogger("run_logger")


# TODO: Make this part of the RandomSampleCrystalPopulation
# Either as static or private method
def create_random_crystal(
    present_species: list[str],
    n_atoms: int,
    closest_distances: CustomClosestDistances,
    possible_space_groups: Iterable[int] = range(1, 231),
    logger: Logger | None = None,
    n_tries_composition: int = 1000,
    seed: int = 42,
    sort_composition_descending: bool = True,
) -> ase.Atoms | None:
    """Worker function to create a random structure."""
    # Set the random seed for reproducibility
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
    n_tries = 0
    while n_tries < n_tries_composition:
        n_tries += 1

        # Ensure that every species is present in the crystal at least once
        n_atoms_per_species = [1 for _ in range(0, n_present_species)]

        # Distribute the remaining atoms randomly
        remaining = n_atoms - n_present_species
        for _ in range(0, remaining):
            index_to_add = rng.integers(0, n_present_species)
            n_atoms_per_species[index_to_add] += 1

            space_group = rng.choice(list(possible_space_groups))

        if sort_composition_descending:
            # Sort the composition in descending order.
            # This can be useful if the present species are sorted in
            # descending order of their guessed appearance in the
            # target features.
            n_atoms_per_species = sorted(n_atoms_per_species, reverse=True)

        # Check if the space group is compatible with the number of atoms
        compatible, _ = Group(space_group, dim=3).check_compatible(n_atoms_per_species)

        if compatible:
            break

    try:
        # Tol_matrix applies an internal factor of 0.5,
        # therefore I need to double the distance here
        tol_matrix = Tol_matrix(
            prototype="atomic", factor=closest_distances.ratio_of_covalent_radii * 2
        )

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
            conventional=True,  # Ensure that set number of atoms is kept
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
        soap_obj: GlobalSOAP,
        target_features: np.ndarray,
        closest_distances: CustomClosestDistances,
        n_atoms,
        fitness_functions: (
            Sequence[FitnessFunction | tuple[FitnessFunction, float]] | FitnessFunction
        ),
        n_samples: int = 1000,
        n_jobs: int = 1,
        exclude_space_groups: Iterable[int] | None = [
            215,
            195,
        ],  # Exclude 215 and 195, since they somehow cause problems
        logger: None | Logger = None,
        rng: None | np.random.Generator = None,
    ):
        if not rng:
            rng = np.random.default_rng()
        self._rng = rng

        self.soap_obj = soap_obj
        self.target_features = target_features
        self.closest_distances = closest_distances
        self.fitness_functions = fitness_functions
        self.n_samples = n_samples
        self.n_jobs = n_jobs
        self.exclude_space_groups = exclude_space_groups
        self.n_atoms = n_atoms
        self.logger = logger

        self.possible_space_groups = self._get_possible_space_groups(n_atoms)

        # Get species present in soap, sorted by their estimated apperance
        self.present_species = global_soap_target.utils.get_present_species(
            soap_obj=soap_obj,
            feature_vector=target_features,
            sort_by_appearance=True,
        )

        # Generate unique seeds for sampling
        # Do this in init so recalling the generate individuals method always
        # creates the same individuals
        self._sample_seeds = self._rng.choice(99999, size=self.n_samples, replace=False)

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
                    create_random_crystal,
                    [
                        (
                            self.present_species,
                            self.n_atoms,
                            self.closest_distances,
                            self.possible_space_groups,
                            None,
                            1000,
                            seed,
                            True,
                        )
                        for seed in self._sample_seeds
                    ],
                )
        else:
            results = [
                create_random_crystal(
                    self.present_species,
                    self.n_atoms,
                    self.closest_distances,
                    self.possible_space_groups,
                    None,
                    1000,
                    seed,
                    True,
                )
                for seed in self._sample_seeds
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
            ind = convert_ase_atoms_to_individual(crystal)
            individuals.append(ind)

        if self.logger:
            self.logger.info(
                f"Generated {len(individuals)} individuals for n_atoms={self.n_atoms}"
            )

        core_utils.fitness_utils.assign_fitness_to_individuals(
            individuals=individuals,
            fitness_functions=self.fitness_functions,
        )

        individuals = population_selections.NSGA2Selection().select(individuals, n)

        # Copy random inds if not enough ind could be generated
        if len(individuals) < n:
            if self.logger:
                self.logger.warning(
                    f"Could only generate {len(individuals)}/{n} "
                    "individuals. Extending population."
                )
            extended_inds = individuals
            while len(extended_inds) <= n:
                idx = self._rng.choice(range(len(extended_inds)))
                extended_inds.append(extended_inds[idx].copy())

        return individuals
