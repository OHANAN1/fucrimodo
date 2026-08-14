# Description: Main script for running the multi-stage search

import logging
import os
import warnings
import random

import numpy as np
from fucrimodo.core import MultiStageSearch
from fucrimodo.core import utils as core_utils
from fucrimodo.utils.target_file_parser import load_target_file
from fucrimodo.core.abstracts import PopulationGenerator
from fucrimodo.core.utils import reproducability as reprod
from fucrimodo.customs import fitness_functions as ff
from fucrimodo.customs import population_generators as pop_gen
from fucrimodo.customs import population_selections as pop_sel
from fucrimodo.customs.ga_stage import (
    GAStage,
    crossovers,
    mutations,
)
from fucrimodo.customs.utils import (
    get_soap_similarity_fitness_list,
    get_species_specific_soap_sim_fitness_list,
    get_n_atoms_from_additional_notes,
)
from fucrimodo.customs import break_conditions

np.random.seed(42)
random.seed(42)

# ── Set random seed ─────────────────────────────────────────────────────
global_rng = np.random.default_rng(42)
warnings.filterwarnings("once")

logging.getLogger()

n_samples = 50  # 5000
n_gen_explore = 1000  # 1000
n_gen_optimize = 500  # 500
n_gen_exploit = 2000  # 2000
N_IND = 50  # 500
n_iterations = 7  # 7


def get_population_generator(
    multi_stage_search: MultiStageSearch,
    closest_distances: core_utils.CustomClosestDistances,
    n_atoms: int,
    n_samples: int = 1000,
    n_jobs: int = 4,
) -> PopulationGenerator:

    fitness_functions = get_soap_similarity_fitness_list(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        n_jobs=n_jobs,
        round_result=None,
    ) + get_species_specific_soap_sim_fitness_list(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        species=multi_stage_search.descriptor_object.species,
        rbf_gamma=0.001,
        n_jobs=n_jobs,
        round_result=None,
    )

    pop_generator = pop_gen.RandomSampleCrystalPopulation(
        soap_obj=multi_stage_search.descriptor_object,
        target_features=multi_stage_search.target_features,
        closest_distances=closest_distances,
        fitness_functions=fitness_functions,
        n_samples=n_samples,
        n_jobs=n_jobs,
        n_atoms=n_atoms,
        logger=multi_stage_search.logger,
        exclude_space_groups=[
            215,
            195,
        ],  # Exclude 215 and 195, since they somehow cause problems
        rng=global_rng,
    )
    return pop_generator


def get_exploration_stage(
    multi_stage_search: MultiStageSearch,
    closest_distances: core_utils.CustomClosestDistances,
    closest_distances_strict: core_utils.CustomClosestDistances,
    cell_bound: core_utils.CustomCellBounds,
    global_break_condition: break_conditions.BreakCondition,
    name: str = "Exploration GA",
    n_generations: int = 500,
    species_specific_fitness_rbf_gamma: float = 1.0,
    n_jobs: int = 1,
) -> GAStage:
    """Generates stage to eplore many structures."""

    fitness_func_list = get_soap_similarity_fitness_list(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        rbf_gammas=[0.1],
        function_titles=["ref_soap_sim"],
        round_result=None,
        n_jobs=n_jobs,
    )
    fitness_func_list += get_species_specific_soap_sim_fitness_list(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        species=multi_stage_search.descriptor_object.species,
        rbf_gamma=species_specific_fitness_rbf_gamma,
        round_result=None,
        n_jobs=n_jobs,
    )
    fitness_func_list += [
        ff.PhysicalityFitness(closest_distances_strict),
    ]

    mutation_list = [
        mutations.elem_mut.PermutationMutation(
            closest_distances=closest_distances, prob=0.1, rng=global_rng
        ),
        mutations.elem_mut.PermutationMutation(
            closest_distances=closest_distances, prob=0.33, rng=global_rng
        ),
        mutations.elem_mut.ReplaceAtomsMutation(
            closest_distances=closest_distances,
            possible_elements=multi_stage_search.descriptor_object.species,
            rng=global_rng,
        ),
        mutations.cell_mut.StrainMutation(
            closest_distances=closest_distances,
            n_variable_cell_vectors=1,
            rng=global_rng,
        ),
        mutations.cell_mut.StrainMutation(
            closest_distances=closest_distances,
            n_variable_cell_vectors=2,
            rng=global_rng,
        ),
        mutations.cell_mut.StrainMutation(
            closest_distances=closest_distances,
            n_variable_cell_vectors=3,
            rng=global_rng,
        ),
        mutations.sym_mut.GetConventionalCellMutation(
            closest_distances=closest_distances,
            symmetry_tol=0.1,
            rng=global_rng,
        ),
        mutations.cell_mut.ScaleUnitCellMutation(
            closest_distances=closest_distances,
            cell_bounds=cell_bound,
            max_scale=1.5,
            min_scale=0.5,
            n_variable_cell_vectors=1,
            rng=global_rng,
        ),
        mutations.cell_mut.ScaleUnitCellMutation(
            closest_distances=closest_distances,
            cell_bounds=cell_bound,
            max_scale=1.2,
            min_scale=0.8,
            n_variable_cell_vectors=2,
            rng=global_rng,
        ),
        mutations.cell_mut.MinimizeTiltMutation(
            closest_distances,
            rng=global_rng,
        ),
    ]
    mutation_list.append(
        mutations.multi_mut.MultipleMutations(
            mutations=mutation_list,
            closest_distances=closest_distances,
            number_of_mutations=2,
            rng=global_rng,
        )
    )

    crossover_list = [
        crossovers.CutAndSpliceCrossover(
            closest_distances=closest_distances,
            cell_bounds=cell_bound,
            rng=global_rng,
        ),
        crossovers.CutAndSpliceCrossover(
            closest_distances=closest_distances,
            cell_bounds=cell_bound,
            number_of_variable_cell_vectors=3,
            rng=global_rng,
        ),
        crossovers.UnitCellCrossover(
            closest_distances=closest_distances,
            rng=global_rng,
        ),
        crossovers.OnePointElementCrossover(
            closest_distances=closest_distances,
            rng=global_rng,
        ),
        crossovers.OnePointPositionCrossover(
            closest_distances=closest_distances,
            rng=global_rng,
        ),
    ]

    break_condition = break_conditions.MultipleOrBreak(
        [
            break_conditions.GenerationBreak(n_generations),
            global_break_condition,
        ]
    )

    parent_selection = pop_sel.TournamentDCDSelection(rng=global_rng)
    survivor_selection = pop_sel.NSGA2Selection()

    return GAStage(
        name=name,
        fitness_functions=fitness_func_list,
        crossover_list=crossover_list,
        mutation_list=mutation_list,
        mutation_probability=0.2,
        crossover_probability=0.8,
        break_condition=break_condition,
        parent_selection=parent_selection,
        survivor_selection=survivor_selection,
        parent_ratio=0.5,
        description="",
        save_n_structures=10,
        rng=global_rng,
    )


def get_optimization_stage(
    multi_stage_search: MultiStageSearch,
    closest_distances: core_utils.CustomClosestDistances,
    closest_distances_strict: core_utils.CustomClosestDistances,
    cell_bound: core_utils.CustomCellBounds,
    global_break_condition: break_conditions.BreakCondition,
    name: str = "Optimization GA",
    n_generations: int = 500,
    n_jobs: int = 1,
) -> GAStage:
    """Generates stage to only slightly optimize structures."""

    fitness_func_list = get_soap_similarity_fitness_list(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        rbf_gammas=[0.1],
        function_titles=["ref_soap_sim"],
        round_result=None,
        n_jobs=n_jobs,
    )
    fitness_func_list += get_species_specific_soap_sim_fitness_list(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        species=multi_stage_search.descriptor_object.species,
        rbf_gamma=0.1,
        round_result=2,
        n_jobs=n_jobs,
    )
    fitness_func_list += [
        ff.PhysicalityFitness(closest_distances_strict),
    ]

    mutation_list = [
        mutations.elem_mut.ReplaceAtomsMutation(
            possible_elements=multi_stage_search.descriptor_object.species,
            closest_distances=closest_distances,
            rng=global_rng,
        ),
        mutations.pos_mut.RattleMutation(
            closest_distances=closest_distances,
            rattle_strength=0.3,
            rattle_prop=0.8,
            rng=global_rng,
        ),
        mutations.pos_mut.RattleMutation(
            closest_distances=closest_distances,
            rattle_strength=0.5,
            rattle_prop=0.8,
            rng=global_rng,
        ),
        mutations.pos_mut.RattleMutation(
            closest_distances=closest_distances,
            rattle_strength=0.5,
            rattle_prop=1.0,
            n_top=1,
            rng=global_rng,
        ),
        mutations.elem_mut.PermutationMutation(
            closest_distances=closest_distances,
            prob=0.3,
            rng=global_rng,
        ),
        mutations.elem_mut.PermutationMutation(
            closest_distances=closest_distances,
            prob=0.1,
            rng=global_rng,
        ),
        mutations.elem_mut.PermutationMutation(
            closest_distances=closest_distances,
            prob=0.01,
            rng=global_rng,
        ),
        mutations.elem_mut.ReplaceAtomsMutation(
            closest_distances=closest_distances,
            possible_elements=multi_stage_search.descriptor_object.species,
            rng=global_rng,
        ),
        mutations.cell_mut.StrainMutation(
            closest_distances=closest_distances,
            n_variable_cell_vectors=3,
            stddev=0.1,
            rng=global_rng,
        ),
        mutations.cell_mut.StrainMutation(
            closest_distances=closest_distances,
            n_variable_cell_vectors=1,
            stddev=0.3,
            rng=global_rng,
        ),
        mutations.sym_mut.GetConventionalCellMutation(
            closest_distances=closest_distances,
            symmetry_tol=0.3,
            rng=global_rng,
        ),
        mutations.pos_mut.MirrorMutation(
            closest_distances,
            rng=global_rng,
        ),
        mutations.cell_mut.RotationMutation(
            closest_distances,
            rng=global_rng,
        ),
        mutations.cell_mut.MinimizeTiltMutation(
            closest_distances,
            rng=global_rng,
        ),
    ]
    mutation_list.append(
        mutations.multi_mut.MultipleMutations(
            mutation_list,
            closest_distances,
            2,
            rng=global_rng,
        )
    )

    crossover_list = [
        crossovers.CutAndSpliceCrossover(
            closest_distances=closest_distances,
            cell_bounds=cell_bound,
            rng=global_rng,
        ),
        crossovers.CutAndSpliceCrossover(
            closest_distances=closest_distances,
            cell_bounds=cell_bound,
            number_of_variable_cell_vectors=3,
            rng=global_rng,
        ),
        crossovers.UnitCellCrossover(
            closest_distances=closest_distances,
            rng=global_rng,
        ),
        crossovers.OnePointElementCrossover(
            closest_distances=closest_distances,
            rng=global_rng,
        ),
        crossovers.OnePointPositionCrossover(
            closest_distances=closest_distances,
            rng=global_rng,
        ),
    ]

    break_condition = break_conditions.MultipleOrBreak(
        [
            break_conditions.GenerationBreak(n_generations),
            global_break_condition,
        ]
    )
    parent_selection = pop_sel.TournamentSelection(3, rng=global_rng)
    survivor_selection = pop_sel.NSGA2Selection()

    return GAStage(
        name=name,
        fitness_functions=fitness_func_list,
        crossover_list=crossover_list,
        mutation_list=mutation_list,
        mutation_probability=0.2,
        crossover_probability=0.5,
        break_condition=break_condition,
        parent_selection=parent_selection,
        survivor_selection=survivor_selection,
        parent_ratio=0.2,
        description="",
        save_n_structures=10,
        rng=global_rng,
    )


def get_exploitation_stage(
    multi_stage_search: MultiStageSearch,
    closest_distances: core_utils.CustomClosestDistances,
    closest_distances_strict: core_utils.CustomClosestDistances,
    cell_bound: core_utils.CustomCellBounds,
    global_break_condition: break_conditions.BreakCondition,
    name: str = "Fine Optimization GA",
    n_generations: int = 100,
    n_jobs: int = 1,
) -> GAStage:
    """Generates stage to only slightly optimize structures."""

    fitness_func_list = get_soap_similarity_fitness_list(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        rbf_gammas=[0.1],
        function_titles=["ref_soap_sim"],
        round_result=None,
        n_jobs=n_jobs,
    )
    fitness_func_list += [
        ff.PhysicalityFitness(closest_distances_strict),
    ]

    mutation_list = [
        mutations.elem_mut.ReplaceAtomsMutation(
            possible_elements=multi_stage_search.descriptor_object.species,
            closest_distances=closest_distances,
            rng=global_rng,
        ),
        mutations.pos_mut.RattleMutation(
            closest_distances=closest_distances,
            rattle_strength=0.001,
            rattle_prop=0.5,
            rng=global_rng,
        ),
        mutations.pos_mut.RattleMutation(
            closest_distances=closest_distances,
            rattle_strength=0.001,
            rattle_prop=1.0,
            n_top=1,
            rng=global_rng,
        ),
        mutations.pos_mut.RattleMutation(
            closest_distances=closest_distances,
            rattle_strength=0.01,
            rattle_prop=1.0,
            n_top=1,
            rng=global_rng,
        ),
        mutations.elem_mut.PermutationMutation(
            closest_distances=closest_distances,
            prob=0.3,
            rng=global_rng,
        ),
        mutations.elem_mut.ReplaceAtomsMutation(
            closest_distances=closest_distances,
            possible_elements=multi_stage_search.descriptor_object.species,
            rng=global_rng,
        ),
        mutations.cell_mut.StrainMutation(
            closest_distances=closest_distances,
            n_variable_cell_vectors=3,
            stddev=0.01,
            rng=global_rng,
        ),
        mutations.cell_mut.StrainMutation(
            closest_distances=closest_distances,
            n_variable_cell_vectors=1,
            stddev=0.01,
            rng=global_rng,
        ),
        mutations.sym_mut.GetConventionalCellMutation(
            closest_distances=closest_distances,
            symmetry_tol=0.2,
            rng=global_rng,
        ),
        mutations.pos_mut.MirrorMutation(
            closest_distances,
            rng=global_rng,
        ),
        mutations.cell_mut.RotationMutation(
            closest_distances,
            rng=global_rng,
        ),
        mutations.cell_mut.MinimizeTiltMutation(
            closest_distances,
            rng=global_rng,
        ),
    ]
    mutation_list.append(
        mutations.multi_mut.MultipleMutations(
            mutation_list,
            closest_distances,
            2,
            rng=global_rng,
        )
    )

    crossover_list = [
        crossovers.CutAndSpliceCrossover(
            closest_distances=closest_distances,
            cell_bounds=cell_bound,
            rng=global_rng,
        ),
        crossovers.UnitCellCrossover(
            closest_distances=closest_distances,
            rng=global_rng,
        ),
        crossovers.OnePointElementCrossover(
            closest_distances=closest_distances,
            rng=global_rng,
        ),
        crossovers.OnePointPositionCrossover(
            closest_distances=closest_distances,
            rng=global_rng,
        ),
    ]

    break_condition = break_conditions.MultipleOrBreak(
        [
            break_conditions.GenerationBreak(n_generations),
            global_break_condition,
        ]
    )
    parent_selection = pop_sel.TournamentSelection(5, rng=global_rng)
    survivor_selection = pop_sel.NSGA2Selection()

    return GAStage(
        name=name,
        fitness_functions=fitness_func_list,
        crossover_list=crossover_list,
        mutation_list=mutation_list,
        mutation_probability=0.5,
        crossover_probability=0.5,
        break_condition=break_condition,
        parent_selection=parent_selection,
        survivor_selection=survivor_selection,
        parent_ratio=0.8,
        description="",
        save_n_structures=10,
        rng=global_rng,
    )


def main(
    name: str | None,
    save_dir: str,
    target_file_path: str,
    n_parallel: int,
    verbose: bool,
    *args,
):
    """Main function to run the multi-stage search."""
    descriptor_obj, target_features, additional_notes = load_target_file(
        target_file_path
    )

    multi_stage_search = MultiStageSearch(
        save_dir=save_dir,
        target_features=np.array(target_features),
        descriptor_object=descriptor_obj,
        descriptive_name=name,
        log_level=logging.INFO,
        n_jobs=n_parallel,
    )
    multi_stage_search.logger.info(f"{multi_stage_search.name}: Starting setup.")

    multi_stage_search.description = (
        f"Last commit msg of lab: {reprod.get_last_commit_msg(os.getcwd())}"
    )
    multi_stage_search.store_file(__file__, "run_config.py")
    multi_stage_search.store_file(target_file_path, "input_file.json")

    if verbose:
        multi_stage_search.logger.setLevel(logging.DEBUG)

    # Check node on which the script runs for debug purposes of run fails
    value = os.environ.get("SLURMD_NODENAME", "Not set")
    if value != "Not set":
        multi_stage_search.logger.info(f"Running on: {value}")
    else:
        multi_stage_search.logger.info("Not running on slurm.")

    # ── Global Setup ───────────────────────────────────────────────────
    soap_species = multi_stage_search.descriptor_object.species

    closest_distances = core_utils.CustomClosestDistances(
        species=soap_species, ratio_of_covalent_radii=0.7
    )

    closest_distances_strict = core_utils.CustomClosestDistances(
        species=soap_species, ratio_of_covalent_radii=1.0
    )

    cell_bound = core_utils.CustomCellBounds(
        {
            "a": [1, 100],
            "b": [1, 100],
            "c": [1, 100],
            "alpha": [10, 170],
            "beta": [10, 170],
            "gamma": [10, 170],
        }
    )

    # ── Extract n atoms from additional notes ──────────────────────
    n_atoms = get_n_atoms_from_additional_notes(additional_notes)
    multi_stage_search.logger.info(
        f"{multi_stage_search.name}: Target atom has {n_atoms} atoms."
    )

    # ── Setup Global Statistics ─────────────────────────────────────────────
    reference_similarity = ff.SoapRbfSimilarityFitness(
        target_soap_features=multi_stage_search.target_features,
        soap_object=multi_stage_search.descriptor_object,
        rbf_gamma=0.1,
        db_title="Reference",
        round_result=None,
    )

    multi_stage_search.global_statistics_dict = {
        "Reference_Similarity": reference_similarity.evaluate_individual,
        "Volume": lambda ind: ind.get_volume(),
    }

    # ── Run Build-up Phase ─────────────────────────────────────────
    population_generator = get_population_generator(
        multi_stage_search=multi_stage_search,
        n_atoms=n_atoms,
        n_samples=n_samples,
        closest_distances=closest_distances,
        n_jobs=multi_stage_search.n_jobs,
    )
    population = population_generator.generate_population(N_IND)

    multi_stage_search.logger.debug(
        f"{multi_stage_search.name}: "
        f"Fitness of population: {population.individuals[0].fitness.values}. "
        f"Number of ind: {len(population.individuals)}"
    )

    # Activate to store first structure
    if True:
        from ase.io import write

        write(
            os.path.join(multi_stage_search.run_dir, "best_0.xsf"),
            population.individuals[0],
        )
        with open(
            os.path.join(multi_stage_search.run_dir, "best_0_data.txt"), "w"
        ) as f:
            f.write(f"Similarities: {population.individuals[0].fitness.values}")

    # ╔══════════════════════════════════════════════════════════╗
    # ║                        Run Stages                        ║
    # ╚══════════════════════════════════════════════════════════╝
    multi_stage_search.logger.info(f"{multi_stage_search.name}: Starting stages.")

    global_break_condition = break_conditions.MaxFitnessBreak(0, 0.99)

    rattle_mut = mutations.pos_mut.RattleMutation(
        closest_distances=closest_distances,
        rattle_strength=0.1,
        rattle_prop=0.5,
        rng=global_rng,
    )
    rattle_mut.logger = logging.getLogger("rattle_mutation")

    for i in range(n_iterations):
        # Rattle inds so they get diversivied and go out of bad local minima
        rattled_inds = []
        for ind in population.individuals:
            new_ind, _ = rattle_mut.mutate(ind.copy())
            rattled_inds.append(new_ind)
        population.individuals = rattled_inds

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
            n_jobs=multi_stage_search.n_jobs,
        )
        multi_stage_search.run(population=population, stage=explore_ga_stage)
        # If good enough individuals are found, break loop
        if global_break_condition.check(population):
            multi_stage_search.logger.info("Found good enough individuals.")
            break

        optimize_ga_stage = get_optimization_stage(
            multi_stage_search=multi_stage_search,
            closest_distances=closest_distances,
            closest_distances_strict=closest_distances_strict,
            cell_bound=cell_bound,
            global_break_condition=global_break_condition,
            name=f"Optimization GA {i}",
            n_generations=n_gen_optimize,
            n_jobs=multi_stage_search.n_jobs,
        )
        multi_stage_search.run(population=population, stage=optimize_ga_stage)
        # If good enough individuals are found, break loop
        if global_break_condition.check(population):
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
        core_utils.fitness_utils.assign_fitness_to_population(
            population, [reference_similarity]
        )
        population.individuals = pop_sel.NSGA2Selection().select(
            population.individuals, 100
        )

        fine_optimize_ga_stage = get_exploitation_stage(
            multi_stage_search=multi_stage_search,
            closest_distances=closest_distances,
            closest_distances_strict=closest_distances_strict,
            cell_bound=cell_bound,
            global_break_condition=global_break_condition,
            name=f"Fine Optimization GA {i}",
            n_generations=n_gen_exploit,
            n_jobs=multi_stage_search.n_jobs,
        )
        multi_stage_search.run(population=population, stage=fine_optimize_ga_stage)

        if global_break_condition.check(population):
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

    # global break was not reached do last optimization
    core_utils.fitness_utils.assign_fitness_to_population(
        population, [reference_similarity]
    )
    if not global_break_condition.check(population):
        multi_stage_search.logger.info("Attemping last optimization.")
        exploitation_ga_stage = get_exploitation_stage(
            multi_stage_search=multi_stage_search,
            closest_distances=closest_distances,
            closest_distances_strict=closest_distances_strict,
            cell_bound=cell_bound,
            global_break_condition=global_break_condition,
            name="Final Optimization GA",
            n_generations=n_gen_exploit,
            n_jobs=multi_stage_search.n_jobs,
        )
        multi_stage_search.run(population=population, stage=exploitation_ga_stage)

    multi_stage_search.logger.info(f"{multi_stage_search.name}: Finished run.")
