# Description: Main script for running the multi-stage search

import logging
import os
import random
import re
import subprocess
import warnings
from collections.abc import Iterable
from io import StringIO
from logging import Logger
from multiprocessing import Pool

import ase
import numpy as np
from ase.build import sort
from ase.geometry import get_distances
from fucrimodo.core import multi_stage_search as multi_stage
from fucrimodo.core.modules import Individual, Population, PopulationGenerator
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.utils.custom_soap import CustomSOAP
from fucrimodo.core.utils.population_utils import (
    assign_fitness_to_all_individuals,
)
from fucrimodo.customs import fitness_functions as ff
from fucrimodo.customs import population_generator as pop_gen
from fucrimodo.customs import population_selections
from fucrimodo.customs.fitness_functions import FitnessFunction
from fucrimodo.customs.ga_stage import (
    GAStage,
    break_conditions,
    crossovers,
    mutations,
)
from fucrimodo.customs.ga_stage.presets import (
    get_soap_similarity_fitness_list,
    get_species_specific_soap_fitness_list,
)
from fucrimodo.utils import soap_similarity as soap_sim
from pyxtal import pyxtal
from pyxtal.symmetry import Group
from pyxtal.tolerance import Tol_matrix

# ── Set random seed ─────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
warnings.filterwarnings("once")


class PhysicalityFitness(FitnessFunction):
    def __init__(
        self,
        closest_distances: dict[tuple[int, int], float],
        db_title: str | None = "PhysicalityFitness",
    ):
        super().__init__(db_title=db_title)
        self.closest_distances = closest_distances

    def __calculate_normalized_atom_distance_fitness(
        self,
        crystal: ase.Atoms,
    ) -> float:
        """
        The Bigger the better.
        Calculates the distances of all atoms in the crystal.
        If the distance between two atoms is bigger or equal to the
        min_allowed_dist the fitness is increased by 1.

        The minimal distance between two atoms is calculated by the
        covalent radii of the atoms with the closest_distances_generator
        function from ase.ga.utilities.

        Nomalized by N(N-1)/2.
        """
        positions = crystal.get_positions()
        atomic_numbers = crystal.get_atomic_numbers()
        cell = crystal.get_cell()

        _, distances = get_distances(p1=positions, cell=cell, pbc=True)

        exponent = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = distances[i, j]
                min_allowed_dist = self.closest_distances[
                    (atomic_numbers[i], atomic_numbers[j])
                ]
                exponent += np.max(
                    [(min_allowed_dist - distance) / min_allowed_dist, 0],
                )

        fitness = np.exp(-exponent)

        return fitness

    def evaluate_individual(self, individual: Individual) -> float:
        return self.__calculate_normalized_atom_distance_fitness(
            crystal=individual,
        )

    def __repr__(self) -> str:
        r_str = "PhysicalityFitness()"
        return r_str


def extract_regex_from_additional_notes(
    additional_notes: str,
    regex: str,
) -> str | None:
    match = re.search(regex, additional_notes)
    if match:
        # If there is a match, return the volume
        return str(match.group(1))
    else:
        raise ValueError(
            f"Could not find the pattern in the additional notes"
            f" with the regex pattern {regex}. \n"
            f"Additional notes: {additional_notes}"
        )


def get_target_individual_from_additional_notes(
    additional_notes: str,
) -> Individual:
    """Load the target structure from the additional notes of the input file.

    It is assumed that the target structure is stored in the additional notes
    as a CIF string. The CIF string is extracted from the additional notes
    using a regex pattern. The CIF string is then loaded into an ASE Atoms
    object.
    """
    cif_string = extract_regex_from_additional_notes(
        additional_notes,
        r"CIF:(.*)",
    )
    assert (
        cif_string is not None
    ), "Could not find CIF string in additional notes."

    cif_string = cif_string.replace("NEWLINE", "\n")
    cif_string = cif_string.replace("QUOTATION_MARK", '"')
    with StringIO(cif_string) as f:
        from ase.io import read

        target_structure = read(f, format="cif")

    assert (
        type(target_structure) is ase.Atoms
    ), "Check if CIF-string is ase.Atoms object"

    target_individual = pop_gen.convert_ase_atoms_to_individual(
        target_structure
    )

    return target_individual


def get_last_commit_msg() -> str:
    commit_msg = ""
    try:
        commit_msg = subprocess.check_output(
            ["git", "-C", f"{os.getcwd()}", "-P", "log", "-1", "--pretty=%B"],
        )
    except Exception as e:
        commit_msg = f"Could not get git status. Error {e}"
        print("Could not get git commit msg.")
    finally:
        return str(commit_msg)


def get_present_species(
    soap_obj: CustomSOAP,
    feature_vector: np.ndarray,
    sort_by_appearance: bool = True,
) -> list[str]:
    """Analyses the species parts of the soap object.

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
    unique_soap_obj_species = list(set(soap_obj.species))

    # Loop over all unique species in the soap object
    species_with_features: list[str] = []
    feature_sum_per_species: list[float] = []
    for single_specie in unique_soap_obj_species:
        # Get the slice obj for the part of the feature vector that
        # corresponds to the current species
        species_slice = soap_obj.get_location((single_specie, single_specie))

        # Calculate the sum of absolute values of the feature vector
        feature_vec_abs_sum = float(
            np.sum(np.abs(feature_vector[species_slice]))
        )

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
        species_with_features = [
            species_with_features[i] for i in sort_indices
        ]

    return species_with_features


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
        compatible, _ = Group(space_group, dim=3).check_compatible(
            n_atoms_per_species
        )
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
            logger.error(
                "Could not create random crystal in an appropriate time."
            )
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
        present_species: list[str],
        target_features: np.ndarray,
        closest_distances: CustomClosestDistances,
        n_samples: int = 1000,
        n_jobs: int = 1,
        rbf_gamma: float = 0.1,
        max_n_atoms: int = 6,
        possible_space_groups: Iterable[int] = range(1, 231),
        rng=np.random.default_rng(),
        logger: None | Logger = None,
        pool=None,
    ):
        self.soap_obj = soap_obj
        self.target_features = target_features
        self.closest_distances = closest_distances
        self.n_samples = n_samples
        self.n_jobs = n_jobs
        self.rbf_similarity = soap_sim.RBFSimilarity(
            target_feature_vector=target_features,
            rbf_gamma=rbf_gamma,
            adjust_gamma=False,
        )
        self.max_n_atoms = max_n_atoms
        self.possible_space_groups = possible_space_groups
        self.present_species = present_species
        self.rng = rng
        self.logger = logger
        self.pool = pool

    def generate_individuals(self, n: int) -> list[Individual]:
        if self.logger is not None:
            self.logger.info(
                f"Creating random crystals with n_atoms={self.max_n_atoms}..."
            )

        results = []
        if self.pool is not None:
            results = self.pool.starmap(
                create_random_structure,
                [
                    (
                        self.present_species,
                        self.max_n_atoms,
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
                        self.max_n_atoms,
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
            self.logger.info(
                f"{self.max_n_atoms}: Finished creating random crystals."
            )

        # Sort out None values, for which no crystal could be created
        random_crystals: list[ase.Atoms] = []
        for i, result in enumerate(results):
            if result is None:
                if self.logger is not None:
                    self.logger.debug(
                        f"{self.max_n_atoms}: "
                        f"Could not create random crystal {i + 1}."
                    )
            else:
                random_crystals.append(result)

        # Convert the crystals to individuals
        individuals = []
        for crystal in random_crystals:
            ind = pop_gen.convert_ase_atoms_to_individual(crystal)
            individuals.append(ind)

        return individuals

    def generate_population(self, size: int) -> Population:
        return Population(individuals=self.generate_individuals(size))


def save_current_script(save_path: str):
    import shutil

    shutil.copy(__file__, save_path)


def run_build_up_phase(
    multi_stage_search: multi_stage.MultiStageSearch,
    population: Population,
    closest_distances: CustomClosestDistances,
    n_atoms: int,
    n_ind_final: int = 100,
    n_samples: int = 1000,
    n_jobs: int = 4,
) -> Population:
    """Runs the build-up phase of the multi-stage search.

    Note! This build-up process does not activate the time tracker and
    is therefore not included in the time tracking.

    :param population: The initial population to start the build-up phase with.
    :param multi_stage_search: The multi-stage search object.
    :param closest_distances: The closest distances object.
    :param n_atoms: The number of atoms in the target structure.
    :param n_ind_final: The number of individuals to keep after the
        build-up phase.
    :param n_samples: The number of samples to generate.
    :param n_jobs: The number of parallel jobs to use for the build-up phase.
    """
    present_species = get_present_species(
        soap_obj=multi_stage_search.descriptor_object,
        feature_vector=multi_stage_search.target_features,
        sort_by_appearance=True,
    )

    with Pool(n_jobs) as pool:
        # Use every space group but the ones that are not compatible with the
        # current number of atoms
        possible_space_groups = range(1, 231)
        possible_space_groups = [
            sg
            for sg in possible_space_groups
            if Group(sg, dim=3).check_compatible(
                [1 for _ in range(0, n_atoms)]
            )[0]
        ]

        # Also exclude 215 and 195, since they somehow cause problems
        multi_stage_search.logger.warning(
            "Warning: Excluding space group 215 and 195."
        )
        possible_space_groups = [
            sg for sg in possible_space_groups if sg not in [215, 195]
        ]

        pop_gen = RandomSampleCrystalPopulation(
            soap_obj=multi_stage_search.descriptor_object,
            present_species=present_species,
            target_features=multi_stage_search.target_features,
            closest_distances=closest_distances,
            n_samples=n_samples,
            n_jobs=n_jobs,
            rbf_gamma=0.1,
            max_n_atoms=n_atoms,
            possible_space_groups=possible_space_groups,
            logger=multi_stage_search.logger,
            pool=pool,
        )
        inds = pop_gen.generate_population(n_samples).individuals
        multi_stage_search.logger.info(
            f"Generated {len(inds)} individuals for n_atoms={n_atoms}"
        )
        unique_inds = []
        for ind in inds:
            ind = sort(ind)
            if ind not in unique_inds:
                unique_inds.append(ind)

        multi_stage_search.logger.info(
            f"Unique individuals: {len(unique_inds)}"
        )

    population = Population(unique_inds)

    similarity_fitnesses = get_soap_similarity_fitness_list(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        n_jobs=n_jobs,
        round_result=None,
    )

    # Check if I should not use this fitness function here
    species_specific_soap_fitnesses = get_species_specific_soap_fitness_list(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        soap_species=multi_stage_search.descriptor_object.species,
        rbf_gamma=0.001,
        n_jobs=n_jobs,
        round_result=None,
    )

    multi_stage_search.logger.info(
        f"Selecting best {n_ind_final} individuals."
    )
    assign_fitness_to_all_individuals(
        population=population,
        fitness_functions=similarity_fitnesses
        + species_specific_soap_fitnesses,
    )
    population.individuals = population_selections.NSGA2Selection().select(
        population.individuals, n_ind_final
    )

    # Copy random inds if not enough ind could be generated
    if len(population.individuals) < n_ind_final:
        multi_stage_search.logger.warning(
            f"Could only generate {len(population.individuals)}/{n_ind_final} "
            "individuals."
        )
        extended_inds = population.individuals
        while len(extended_inds) <= n_ind_final:
            choosen_ind = random.choice(extended_inds).copy()
            extended_inds.append(choosen_ind)

    return population


def get_exploration_stage(
    multi_stage_search: multi_stage.MultiStageSearch,
    closest_distances: CustomClosestDistances,
    closest_distances_strict: CustomClosestDistances,
    cell_bound: CustomCellBounds,
    global_break_condition: break_conditions.BreakCondition,
    name: str = "Exploration GA",
    n_generations: int = 500,
    species_specific_fitness_rbf_gamma: float = 1.0,
) -> GAStage:
    """Generates stage to eplore many structures."""

    fitness_func_list = get_soap_similarity_fitness_list(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        rbf_gammas=[0.1],
        function_titles=["ref_soap_sim"],
        round_result=None,
    )
    fitness_func_list += get_species_specific_soap_fitness_list(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        soap_species=multi_stage_search.descriptor_object.species,
        rbf_gamma=species_specific_fitness_rbf_gamma,
        round_result=None,
    )
    fitness_func_list += [
        ff.AgeFitness(gamma=0.0001),
        PhysicalityFitness(closest_distances_strict),
    ]

    mutation_list = [
        mutations.elem_mut.PermutationMutation(
            closest_distances=closest_distances, prob=0.1
        ),
        mutations.elem_mut.PermutationMutation(
            closest_distances=closest_distances, prob=0.33
        ),
        mutations.elem_mut.ReplaceAtomsMutation(
            closest_distances=closest_distances,
            possible_elements=multi_stage_search.descriptor_object.species,
        ),
        mutations.cell_mut.StrainMutation(
            closest_distances=closest_distances,
            n_variable_cell_vectors=1,
        ),
        mutations.cell_mut.StrainMutation(
            closest_distances=closest_distances,
            n_variable_cell_vectors=2,
        ),
        mutations.cell_mut.StrainMutation(
            closest_distances=closest_distances,
            n_variable_cell_vectors=3,
        ),
        mutations.sym_mut.GetConventionalCellMutation(
            closest_distances=closest_distances, symmetry_tol=0.1
        ),
        mutations.cell_mut.ScaleUnitCellMutation(
            closest_distances=closest_distances,
            cell_bounds=cell_bound,
            max_scale=1.5,
            min_scale=0.5,
            n_variable_cell_vectors=1,
        ),
        mutations.cell_mut.ScaleUnitCellMutation(
            closest_distances=closest_distances,
            cell_bounds=cell_bound,
            max_scale=1.2,
            min_scale=0.8,
            n_variable_cell_vectors=2,
        ),
        mutations.cell_mut.MinimizeTiltMutation(closest_distances),
    ]
    mutation_list.append(
        mutations.multi_mut.MultipleMutations(
            mutation_list, closest_distances, 2
        )
    )

    crossover_list = [
        crossovers.CutAndSpliceCrossover(
            closest_distances=closest_distances, cell_bounds=cell_bound
        ),
        crossovers.CutAndSpliceCrossover(
            closest_distances=closest_distances,
            cell_bounds=cell_bound,
            number_of_variable_cell_vectors=3,
        ),
        crossovers.UnitCellCrossover(closest_distances=closest_distances),
        crossovers.OnePointElementCrossover(
            closest_distances=closest_distances
        ),
        crossovers.OnePointPositionCrossover(
            closest_distances=closest_distances
        ),
    ]

    break_condition = break_conditions.MultipleOrBreak(
        [
            break_conditions.GenerationBreak(n_generations),
            global_break_condition,
        ]
    )

    return GAStage(
        name=name,
        fitness_functions=fitness_func_list,
        crossover_list=crossover_list,
        mutation_list=mutation_list,
        mutation_probability=0.2,
        crossover_probability=0.8,
        break_condition=break_condition,
        parent_selection=population_selections.TournamentDCDSelection(),
        survivor_selection=population_selections.NSGA2Selection(),
        parent_ratio=0.5,
        description="",
        save_n_crystals=10,
    )


def get_optimization_stage(
    multi_stage_search: multi_stage.MultiStageSearch,
    closest_distances: CustomClosestDistances,
    closest_distances_strict: CustomClosestDistances,
    cell_bound: CustomCellBounds,
    global_break_condition: break_conditions.BreakCondition,
    name: str = "Optimization GA",
    n_generations: int = 500,
) -> GAStage:
    """Generates stage to only slightly optimize structures."""

    fitness_func_list = get_soap_similarity_fitness_list(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        rbf_gammas=[0.1],
        function_titles=["ref_soap_sim"],
        round_result=None,
    )
    fitness_func_list += get_species_specific_soap_fitness_list(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        soap_species=multi_stage_search.descriptor_object.species,
        rbf_gamma=0.1,
        round_result=2,
    )
    fitness_func_list += [
        PhysicalityFitness(closest_distances_strict),
    ]

    mutation_list = [
        mutations.elem_mut.ReplaceAtomsMutation(
            possible_elements=multi_stage_search.descriptor_object.species,
            closest_distances=closest_distances,
            max_steps=100,
        ),
        mutations.pos_mut.RattleMutation(
            closest_distances=closest_distances,
            rattle_strength=0.3,
            rattle_prop=0.8,
        ),
        mutations.pos_mut.RattleMutation(
            closest_distances=closest_distances,
            rattle_strength=0.5,
            rattle_prop=0.8,
        ),
        mutations.pos_mut.RattleMutation(
            closest_distances=closest_distances,
            rattle_strength=0.5,
            rattle_prop=1.0,
            n_top=1,
        ),
        mutations.elem_mut.PermutationMutation(
            closest_distances=closest_distances, prob=0.3
        ),
        mutations.elem_mut.PermutationMutation(
            closest_distances=closest_distances, prob=0.1
        ),
        mutations.elem_mut.PermutationMutation(
            closest_distances=closest_distances, prob=0.01
        ),
        mutations.elem_mut.ReplaceAtomsMutation(
            closest_distances=closest_distances,
            possible_elements=multi_stage_search.descriptor_object.species,
        ),
        mutations.cell_mut.StrainMutation(
            closest_distances=closest_distances,
            n_variable_cell_vectors=3,
            stddev=0.1,
        ),
        mutations.cell_mut.StrainMutation(
            closest_distances=closest_distances,
            n_variable_cell_vectors=1,
            stddev=0.3,
        ),
        mutations.sym_mut.GetConventionalCellMutation(
            closest_distances=closest_distances, symmetry_tol=0.3
        ),
        mutations.pos_mut.MirrorMutation(closest_distances),
        mutations.cell_mut.RotationMutation(closest_distances),
        mutations.cell_mut.MinimizeTiltMutation(closest_distances),
    ]
    mutation_list.append(
        mutations.multi_mut.MultipleMutations(
            mutation_list, closest_distances, 2
        )
    )

    crossover_list = [
        crossovers.CutAndSpliceCrossover(
            closest_distances=closest_distances, cell_bounds=cell_bound
        ),
        crossovers.CutAndSpliceCrossover(
            closest_distances=closest_distances,
            cell_bounds=cell_bound,
            number_of_variable_cell_vectors=3,
        ),
        crossovers.UnitCellCrossover(closest_distances=closest_distances),
        crossovers.OnePointElementCrossover(
            closest_distances=closest_distances
        ),
        crossovers.OnePointPositionCrossover(
            closest_distances=closest_distances
        ),
    ]

    break_condition = break_conditions.MultipleOrBreak(
        [
            break_conditions.GenerationBreak(n_generations),
            global_break_condition,
        ]
    )

    return GAStage(
        name=name,
        fitness_functions=fitness_func_list,
        crossover_list=crossover_list,
        mutation_list=mutation_list,
        mutation_probability=0.2,
        crossover_probability=0.5,
        break_condition=break_condition,
        parent_selection=population_selections.TournamentSelection(3),
        survivor_selection=population_selections.NSGA2Selection(),
        parent_ratio=0.2,
        description="",
        save_n_crystals=10,
    )


def get_fine_optimization_stage(
    multi_stage_search: multi_stage.MultiStageSearch,
    closest_distances: CustomClosestDistances,
    closest_distances_strict: CustomClosestDistances,
    cell_bound: CustomCellBounds,
    global_break_condition: break_conditions.BreakCondition,
    name: str = "Fine Optimization GA",
    n_generations: int = 100,
) -> GAStage:
    """Generates stage to only slightly optimize structures."""

    fitness_func_list = get_soap_similarity_fitness_list(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        rbf_gammas=[0.1],
        function_titles=["ref_soap_sim"],
        round_result=None,
    )
    fitness_func_list += [
        PhysicalityFitness(closest_distances_strict),
    ]

    mutation_list = [
        mutations.elem_mut.ReplaceAtomsMutation(
            possible_elements=multi_stage_search.descriptor_object.species,
            closest_distances=closest_distances,
            max_steps=100,
        ),
        mutations.pos_mut.RattleMutation(
            closest_distances=closest_distances,
            rattle_strength=0.001,
            rattle_prop=0.5,
        ),
        mutations.pos_mut.RattleMutation(
            closest_distances=closest_distances,
            rattle_strength=0.001,
            rattle_prop=1.0,
            n_top=1,
        ),
        mutations.pos_mut.RattleMutation(
            closest_distances=closest_distances,
            rattle_strength=0.01,
            rattle_prop=1.0,
            n_top=1,
        ),
        mutations.elem_mut.PermutationMutation(
            closest_distances=closest_distances, prob=0.3
        ),
        mutations.elem_mut.ReplaceAtomsMutation(
            closest_distances=closest_distances,
            possible_elements=multi_stage_search.descriptor_object.species,
        ),
        mutations.cell_mut.StrainMutation(
            closest_distances=closest_distances,
            n_variable_cell_vectors=3,
            stddev=0.01,
        ),
        mutations.cell_mut.StrainMutation(
            closest_distances=closest_distances,
            n_variable_cell_vectors=1,
            stddev=0.01,
        ),
        mutations.sym_mut.GetConventionalCellMutation(
            closest_distances=closest_distances, symmetry_tol=0.2
        ),
        mutations.pos_mut.MirrorMutation(closest_distances),
        mutations.cell_mut.RotationMutation(closest_distances),
        mutations.cell_mut.MinimizeTiltMutation(closest_distances),
    ]
    mutation_list.append(
        mutations.multi_mut.MultipleMutations(
            mutation_list, closest_distances, 2
        )
    )

    crossover_list = [
        crossovers.CutAndSpliceCrossover(
            closest_distances=closest_distances, cell_bounds=cell_bound
        ),
        crossovers.UnitCellCrossover(closest_distances=closest_distances),
        crossovers.OnePointElementCrossover(
            closest_distances=closest_distances
        ),
        crossovers.OnePointPositionCrossover(
            closest_distances=closest_distances
        ),
    ]

    break_condition = break_conditions.MultipleOrBreak(
        [
            break_conditions.GenerationBreak(n_generations),
            global_break_condition,
        ]
    )

    return GAStage(
        name=name,
        fitness_functions=fitness_func_list,
        crossover_list=crossover_list,
        mutation_list=mutation_list,
        mutation_probability=0.5,
        crossover_probability=0.5,
        break_condition=break_condition,
        parent_selection=population_selections.TournamentSelection(5),
        survivor_selection=population_selections.NSGA2Selection(),
        parent_ratio=0.8,
        description="",
        save_n_crystals=10,
    )


def main(multi_stage_search: multi_stage.MultiStageSearch, additional_notes):
    """Main function to run the multi-stage search."""
    multi_stage_search.logger.info(
        f"{multi_stage_search.name}: Starting setup."
    )

    last_commit_msg = get_last_commit_msg()
    multi_stage_search.description = (
        f"Last commit msg of lab: {last_commit_msg}"
    )
    multi_stage_search.log_level = logging.INFO

    N_IND = 500

    # Set the default number of processes
    N_PARALLEL_STAGES = 4

    # Check if the number of processes is set in the environment when
    # running on a cluster
    value = os.environ.get("SLURM_CPUS_PER_TASK", "Not set")
    if value != "Not set":
        N_PARALLEL_STAGES = int(value)

    # Check node on which the script runs for debug purposes of run fails
    value = os.environ.get("SLURMD_NODENAME", "Not set")
    if value != "Not set":
        multi_stage_search.logger.info(f"Running on: {value}")
    else:
        multi_stage_search.logger.info("Not running on slurm.")

    save_current_script(
        os.path.join(multi_stage_search.run_dir, "run_config.py")
    )

    # ── Global Setup ───────────────────────────────────────────────────
    soap_species = multi_stage_search.descriptor_object.species

    closest_distances = CustomClosestDistances(
        species=soap_species, ratio_of_covalent_radii=0.7
    )

    closest_distances_strict = CustomClosestDistances(
        species=soap_species, ratio_of_covalent_radii=1.0
    )

    cell_bound = CustomCellBounds(
        {
            "a": [1, 100],
            "b": [1, 100],
            "c": [1, 100],
            "alpha": [10, 170],
            "beta": [10, 170],
            "gamma": [10, 170],
        }
    )

    # ── Extract target structure from additional notes ──────────────────────
    target_individual = get_target_individual_from_additional_notes(
        additional_notes
    )
    n_atoms = len(target_individual)
    multi_stage_search.logger.info(
        f"{multi_stage_search.name}: Target atom has {n_atoms} atoms."
    )

    # ── Setup Global Statistics ─────────────────────────────────────────────
    reference_similarity = ff.SimilarityToTargetSOAPFitness(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        soap_similarity=soap_sim.RBFSimilarity(
            target_feature_vector=multi_stage_search.target_features,
            rbf_gamma=0.1,
            adjust_gamma=False,
        ),
        db_title="Reference",
        round_result=None,
    )

    multi_stage_search.global_statistics_dict = {
        "Reference_Similarity": reference_similarity.evaluate_individual,
    }

    # ── Run Build-up Phase ─────────────────────────────────────────
    n_samples = 5000
    population = run_build_up_phase(
        population=Population([]),
        multi_stage_search=multi_stage_search,
        n_atoms=n_atoms,
        n_samples=n_samples,
        n_ind_final=N_IND,
        closest_distances=closest_distances,
        n_jobs=N_PARALLEL_STAGES,
    )

    multi_stage_search.logger.debug(
        f"{multi_stage_search.name}: "
        f"Fitness of population: {population.individuals[0].fitness.values}. "
        f"Number of ind: {len(population.individuals)}"
    )

    # ╔══════════════════════════════════════════════════════════╗
    # ║                        Run Stages                        ║
    # ╚══════════════════════════════════════════════════════════╝
    multi_stage_search.logger.info(
        f"{multi_stage_search.name}: Starting stages."
    )

    global_break_condition = break_conditions.MaxFitnessBreak(0, 0.99)

    rattle_mut = mutations.pos_mut.RattleMutation(
        closest_distances=closest_distances,
        rattle_strength=0.05,
        rattle_prop=0.5,
    )
    rattle_mut.logger = logging.getLogger("rattle_mutation")

    n_stages = 7
    for i in range(n_stages):
        # Rattle inds so they get diversivied and go out of bad local minima
        rattled_inds = []
        for ind in population.individuals:
            new_ind, _ = rattle_mut.mutate(ind.copy())
            rattled_inds.append(new_ind)
        population.individuals = rattled_inds

        n_gen_explore = 1000
        species_spec_fit_rbf_gamma = 0.5
        if i == 0:
            species_spec_fit_rbf_gamma = 0.001

        explore_ga_stage = get_exploration_stage(
            multi_stage_search=multi_stage_search,
            closest_distances=closest_distances,
            closest_distances_strict=closest_distances_strict,
            cell_bound=cell_bound,
            global_break_condition=global_break_condition,
            name=f"Exploration GA {i}",
            n_generations=n_gen_explore,
            species_specific_fitness_rbf_gamma=species_spec_fit_rbf_gamma,
        )
        multi_stage_search.run(population=population, stage=explore_ga_stage)
        # If good enough individuals are found, break loop
        if global_break_condition.check(population.individuals, 1):
            multi_stage_search.logger.info("Found good enough individuals.")
            break

        optimize_ga_stage = get_optimization_stage(
            multi_stage_search=multi_stage_search,
            closest_distances=closest_distances,
            closest_distances_strict=closest_distances_strict,
            cell_bound=cell_bound,
            global_break_condition=global_break_condition,
            name=f"Optimization GA {i}",
            n_generations=500,
        )
        multi_stage_search.run(population=population, stage=optimize_ga_stage)
        # If good enough individuals are found, break loop
        if global_break_condition.check(population.individuals, 1):
            multi_stage_search.logger.info("Found good enough individuals.")
            break

        # Perform fine optimization on part of structures to check if they can
        # be optimized further, if not delete and continue with old population
        # to avoid getting stuck in minimas
        multi_stage_search.logger.info(
            "Selecting 100 individuals for fine optimization."
        )
        # Copy all original individuals from the population to keep them
        original_inds = [ind.copy() for ind in population.individuals]

        # Only select 100 inds for fine optimization
        assign_fitness_to_all_individuals(population, [reference_similarity])
        population.individuals = population_selections.NSGA2Selection().select(
            population.individuals, 100
        )

        fine_optimize_ga_stage = get_fine_optimization_stage(
            multi_stage_search=multi_stage_search,
            closest_distances=closest_distances,
            closest_distances_strict=closest_distances_strict,
            cell_bound=cell_bound,
            global_break_condition=global_break_condition,
            name=f"Fine Optimization GA {i}",
            n_generations=1000,
        )
        multi_stage_search.run(
            population=population, stage=fine_optimize_ga_stage
        )

        if global_break_condition.check(population.individuals, 1):
            # If good enough individuals are found, use fine optimized
            # population
            multi_stage_search.logger.info(
                "Fine optimization found good inds, using them as population."
            )
            del original_inds
            break
        else:
            # If no good ones were found use old individuals to avoid using
            # localy optimized results and get stuck
            multi_stage_search.logger.info(
                "Fine optimization did not find good inds. Overwriting "
                "population with old individuals."
            )
            population.individuals = original_inds

    # gloabal break was not reached do last optimization
    assign_fitness_to_all_individuals(population, [reference_similarity])
    if not global_break_condition.check(population.individuals, 1):
        multi_stage_search.logger.info("Attemping last optimization.")
        fine_optimize_ga_stage = get_fine_optimization_stage(
            multi_stage_search=multi_stage_search,
            closest_distances=closest_distances,
            closest_distances_strict=closest_distances_strict,
            cell_bound=cell_bound,
            global_break_condition=global_break_condition,
            name="Final Optimization GA",
            n_generations=500,
        )
        multi_stage_search.run(
            population=population, stage=fine_optimize_ga_stage
        )

    multi_stage_search.logger.info(f"{multi_stage_search.name}: Finished run.")
