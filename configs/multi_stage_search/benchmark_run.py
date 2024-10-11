# Description: Main script for running the multi-stage search
import numpy as np
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.utils.custom_soap import CustomSOAP
from fucrimodo.customs import population_selections as pop_sel
import random
from fucrimodo.core import multi_stage_search as multi_stage
from fucrimodo.customs.ga_stage import break_conditions as break_cond
import numpy as np
import warnings
import logging
from copy import deepcopy
from fucrimodo.customs.ga_stage import mutations as mut
from fucrimodo.customs.ga_stage import crossovers as cross
from fucrimodo.core.modules import FitnessFunction
from fucrimodo.utils import soap_similarity as soap_sim
from collections.abc import Sequence
from fucrimodo.customs import fitness_functions as ff
from fucrimodo.core.modules import Individual
from fucrimodo.core.modules import Population
from fucrimodo.customs import population_generator as crystal_creation

def get_start_pop_candidates(
        soap_species: list[str],
        population_size: int
    ) -> Population:

    cell_bounds = CustomCellBounds({
        "a": [1, 4], "b": [1, 4], "c": [1, 4], 
        "alpha": [20, 160], "beta": [20, 160], "gamma": [20, 160]
    })

    start_pop_candidates = crystal_creation.create_one_atomic_crystals(
        atom_types=soap_species,
        cell_bounds=cell_bounds,
        total_number_of_atoms=population_size,
    )

    individual_list = []
    for atoms in start_pop_candidates:
        individual_list.append(
            Individual(
                symbols=atoms.get_chemical_symbols(),
                positions=atoms.get_positions(),
                cell=atoms.cell,
                pbc=atoms.pbc
            )
        )

    population = Population(individual_list)

    return population


def get_enlarge_free_mutations(
    closest_distances: CustomClosestDistances,
    soap_species: list[str]
):
    perm_mut = mut.elem_mut.PermutationMutation(
        closest_distances=closest_distances,
    )
    rattle_mut_default = mut.pos_mut.RattleMutation(
        closest_distances=closest_distances,
    )
    rattle_mut_weak = mut.pos_mut.RattleMutation(
        closest_distances=closest_distances, n_top=1, rattle_strength=0.1
    )
    add_mut = mut.elem_mut.AddAtomsMutation(
        possible_elements=soap_species,
        closest_distances=closest_distances
    )
    del_mut = mut.elem_mut.DeleteRandomAtomsMutation(
        closest_distances=closest_distances
    )
    replace_mut = mut.elem_mut.ReplaceAtomsMutation(
        possible_elements=soap_species,
        closest_distances=closest_distances
    )
    # soft_mut = mut.energy_mut.SoftMutation(
    #     closest_distances=closest_distances
    # )
    min_tilt_mut = mut.cell_mut.MinimizeTiltMutation(
        closest_distances=closest_distances
    )
    conv_cell_mut = mut.sym_mut.GetConventionalCellMutation(
        closest_distances=closest_distances,
        symmetry_tol=0.3
    )
    rot_mut = mut.cell_mut.RotationMutation(
        closest_distances=closest_distances
    )
    enlarge_free_muts = [  # noqa
        perm_mut,
        rattle_mut_default,
        rattle_mut_weak,
        add_mut,
        del_mut,
        replace_mut,
        # soft_mut,
        min_tilt_mut,
        conv_cell_mut,
        rot_mut,
    ]
    enlarge_free_multi_mut = mut.multi_mut.MultipleMutations(
        mutations=enlarge_free_muts,
        number_of_mutations=2,
        random_order=True,
        closest_distances=closest_distances
    )
    return enlarge_free_muts, enlarge_free_multi_mut


def get_enlarge_mutations(
    closest_distances: CustomClosestDistances,
    cell_bounds: CustomCellBounds
):
    enlarge_mut = mut.cell_mut.EnlargeMutation(
        closest_distances=closest_distances,
        cell_bounds=cell_bounds
    )
    strain_mut = mut.cell_mut.StrainMutation(
        closest_distances=closest_distances,
        n_variable_cell_vectors=3,
        cell_bounds=cell_bounds,
    )
    cutout_mut = mut.cell_mut.CutoutMutation(
        closest_distances=closest_distances,
        cell_bounds=cell_bounds,
        tolerance=1.
    )
    return [enlarge_mut, strain_mut, cutout_mut]


def get_soap_similarity_fitness_list(
    target_soap_features,
    soap_object: CustomSOAP,
    rbf_gammas: Sequence[float | int] = [1., 0.1, 0.01],
    function_titles: list[str] = [
        "soap_similarity_strong",
        "soap_similarity_mid",
        "soap_similarity_weak"
    ]
) -> list[FitnessFunction]:

    assert len(function_titles) == len(rbf_gammas), "Define same number of titles as rbf gammas."

    soap_fitnesses = []
    for i in range(len(rbf_gammas)):
        soap_fitnesses.append(
            ff.SimilarityToTargetSOAPFitness(
                target_soap_features=target_soap_features,
                soap_object=soap_object,
                soap_similarity=soap_sim.RBFSimilarity(
                    target_feature_vector=target_soap_features,
                    rbf_gamma=rbf_gammas[i],
                    adjust_gamma=False,
                ),
                db_title=function_titles[i]
            )
        )

    return soap_fitnesses


def get_species_specific_soap_fitness_list(
    target_soap_features,
    soap_species: Sequence[str | int],
    soap_object: CustomSOAP,
    rbf_gamma: int | float = 0.1,
    function_name: str = "species_specific_fit"
    ) -> list[FitnessFunction] :
    species_specific_fitnesses= []
    for i in range(len(soap_species)):
        for j in range(i, len(soap_species)):
            soap_fit_spec = ff.SimilarityToTargetSOAPFitness(
                target_soap_features=target_soap_features,
                soap_object=soap_object,
                soap_similarity=soap_sim.SpeciesSpecificRBFSim(
                    target_feature_vector=target_soap_features,
                    rbf_gamma=rbf_gamma,
                    adjust_gamma=False,
                    soap_object=soap_object,
                    species=soap_species[i],  # type: ignore
                    species_to_compare=[soap_species[j]],  # type: ignore
                ),
                db_title="{}_{}_{}".format(
                    function_name, soap_species[i], soap_species[j]
                )
            )
            species_specific_fitnesses.append(soap_fit_spec)

    return species_specific_fitnesses


def main(
    target_features: np.ndarray,
    soap_obj: CustomSOAP,
    log_level: int = logging.INFO,
    warnings_enable: bool = True,
    verbose: int = 3,
    random_seed: int = 42,
    ):

    # ── Set random seed ─────────────────────────────────────────────────────
    random.seed(random_seed)
    np.random.seed(random_seed)

    # ── Set up debugging and warnings ---------------------------------------
    if not warnings_enable:
        warnings.filterwarnings("ignore")

    # ── Global Setup ───────────────────────────────────────────────────
    soap_species = soap_obj.species

    cell_bounds = []
    for l_max in [8, 14]:
        cell_bounds.append(
            CustomCellBounds({
                "a": [1, l_max], "b": [1, l_max], "c": [1, l_max], 
                "alpha": [20, 160], "beta": [20, 160], "gamma": [20, 160],
                "phi": [0, 180], "chi": [0, 180], "psi": [0, 180]
            })
        )

    # ── Start Population Candidates ─────────────────────────────────────────
    population = get_start_pop_candidates(
        soap_species=soap_species,
        population_size=500
    )

    closest_distances = CustomClosestDistances(
        species=soap_species,
        ratio_of_covalent_radii=0.8
    )


    # ── Fitness functions ───────────────────────────────────────────────────
    species_specific_fitnesses = get_species_specific_soap_fitness_list(
        target_soap_features=target_features,
        soap_species=soap_species,
        soap_object=soap_obj,
        rbf_gamma=0.01
    )
    soap_fitness_list = get_soap_similarity_fitness_list(
        target_soap_features=target_features,
        soap_object=soap_obj
    )
    soap_fitness_strong = soap_fitness_list[0]
    soap_fitness_mid = soap_fitness_list[1]
    soap_fitness_weak = soap_fitness_list[2]


    # ── Mutations ───────────────────────────────────────────────────────────
    enlarge_free_muts, enlarge_free_multi_mut = get_enlarge_free_mutations(
        closest_distances, soap_species
    )

    enlarge_muts_1 = get_enlarge_mutations(
        closest_distances, cell_bounds[0]
    )
    enlarge_multi_mut_1 = mut.multi_mut.MultipleMutations(
        mutations= [enlarge_muts_1[0], enlarge_free_multi_mut],
        number_of_mutations=2,
        random_order=False,
        closest_distances=closest_distances
    )
    all_muts_1 = enlarge_free_muts + enlarge_muts_1 + \
        [enlarge_free_multi_mut, enlarge_multi_mut_1]

    enlarge_muts_2 = get_enlarge_mutations(
        closest_distances, cell_bounds[1]
    )
    enlarge_multi_mut_2 = mut.multi_mut.MultipleMutations(
        mutations= [enlarge_muts_2[0], enlarge_free_multi_mut],
        number_of_mutations=2,
        random_order=False,
        closest_distances=closest_distances
    )
    all_muts_2 = enlarge_free_muts + enlarge_muts_2 + \
        [enlarge_free_multi_mut, enlarge_multi_mut_2]


    optimize_mutations = [
        mut.elem_mut.PermutationMutation(closest_distances=closest_distances),
        mut.pos_mut.RattleMutation(
            closest_distances=closest_distances, n_top=1, rattle_strength=0.1
        ),
        mut.pos_mut.RattleMutation(
            closest_distances=closest_distances, n_top=1, rattle_strength=0.5
        ),
        mut.pos_mut.RattleMutation(
            closest_distances=closest_distances, n_top=1, rattle_strength=0.01
        ),
        mut.sym_mut.GetConventionalCellMutation(
            closest_distances=closest_distances,
            symmetry_tol=0.3
        )
    ]
    multi_opti_mut = mut.multi_mut.MultipleMutations(
        mutations=optimize_mutations,
        number_of_mutations=2,
        random_order=True,
        closest_distances=closest_distances
    )
    all_opti_muts = optimize_mutations + [multi_opti_mut]

    soap_fitness_weak = soap_fitness_list[0]
    soap_fitness_mid = soap_fitness_list[1]
    soap_fitness_strong = soap_fitness_list[2]

    # ── Defaults ────────────────────────────────────────────────────────────
    selection_defaults = {
        "survivor_selection": pop_sel.NSGA2Selection(),
        "parent_selection": pop_sel.TournamentSelection(tournament_size=4)
    }

    exploration_defaults = {
        "crossover_probability": 0.8,
        "mutation_probability": 0.8,
        "break_condition": break_cond.MultipleOrBreak([
            break_cond.GenerationBreak(200), 
            break_cond.MaxFitnessBreak(0, 0.80),
            break_cond.MultipleAndBreak([
                break_cond.GenerationBreak(100),
                break_cond.NotBreak(break_cond.MaxFitnessBreak(0, 0.70))
            ])
        ]),
        "crossover_list": [
            cross.OnePointElementCrossover(closest_distances),
            cross.UnitCellCrossover(closest_distances),
        ],
        "fitness_functions": [
             soap_fitness_mid, soap_fitness_weak, soap_fitness_strong # Add mid as first fitness function, since the break condition is based on it
        ] + species_specific_fitnesses,
        **selection_defaults
    }
    optimization_defaults = {
        "crossover_probability": 0.8,
        "mutation_probability": 0.8,
        "break_condition": break_cond.MultipleOrBreak([
            break_cond.GenerationBreak(200),
            break_cond.MaxFitnessBreak(0, 0.99),
            break_cond.MultipleAndBreak([
                break_cond.GenerationBreak(100),
                break_cond.NotBreak(break_cond.MaxFitnessBreak(0, 0.95))
            ])
        ]),
        "crossover_list": [cross.OnePointElementCrossover(closest_distances)],
        "fitness_functions": [
            soap_fitness_mid, soap_fitness_weak, soap_fitness_strong 
        ],
        "mutation_list": all_opti_muts,
        **selection_defaults
    }

    # ╔══════════════════════════════════════════════════════════╗
    # ║                      Initialize Run                      ║
    # ╚══════════════════════════════════════════════════════════╝

    ref_fitness = deepcopy(soap_fitness_mid)
    global_stats_dict = {
        "reference_similarity": ref_fitness.evaluate_individual,
        "volume": lambda x: x.get_volume(),
    }

    multi_stage_search = multi_stage.MultiStageSearch(
        save_dir="data/processed/results/",
        description="Like before, but without Soft mutation, with default rattle in enlarge free Mutations and weaker rattle in optimize mutations.",
        global_statistics_dict=global_stats_dict,
        log_level=log_level
    )

    # ╔══════════════════════════════════════════════════════════╗
    # ║                        Run Stages                        ║
    # ╚══════════════════════════════════════════════════════════╝

    from fucrimodo.customs.ga_stage import GAStage

    # Stage 1: Exploration
    exploration_defaults["crossover_list"].append(
        cross.StackCellsCrossover(closest_distances, cell_bounds[0])
    )
    exploration_defaults["mutation_list"] = all_muts_1
    population = multi_stage_search.run(
        population=population,
        stage = GAStage("Exploration", **exploration_defaults)
    )

    # Stage 2: Optimization
    population = multi_stage_search.run(
        population=population,
        stage = GAStage("Optimization", **optimization_defaults)
    )

    # Stage 3: Extended Exploration
    exploration_defaults["crossover_list"][-1] = cross.StackCellsCrossover(
        closest_distances, cell_bounds[1]
    )
    exploration_defaults["mutation_list"] = all_muts_2
    population = multi_stage_search.run(
        population=population,
        stage = GAStage(f"Extended Exploration", **exploration_defaults)
    )

    # Stage 4: Optimization
    population = multi_stage_search.run(
        population=population,
        stage = GAStage("Optimization", **optimization_defaults)
    )

    multi_stage_search.save_results()
