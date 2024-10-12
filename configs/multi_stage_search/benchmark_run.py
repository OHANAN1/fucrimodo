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
from fucrimodo.customs.ga_stage.presets import ExlorationGAPreset

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


    # ── Mutations ───────────────────────────────────────────────────────────

    optimize_mutations = [
        mut.elem_mut.PermutationMutation(closest_distances=closest_distances),
        mut.pos_mut.RattleMutation(
            closest_distances=closest_distances, n_top=1, rattle_strength=0.1
        ),
        mut.pos_mut.RattleMutation(
            closest_distances=closest_distances, n_top=1, rattle_strength=0.5
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
    all_opti_muts = optimize_mutations # + [multi_opti_mut]

    # ── Defaults ────────────────────────────────────────────────────────────
    selection_defaults = {
        "survivor_selection": pop_sel.NSGA2Selection(),
        "parent_selection": pop_sel.TournamentSelection(tournament_size=4)
    }

    # optimization_defaults = {
    #     "crossover_probability": 0.8,
    #     "mutation_probability": 0.8,
    #     "break_condition": break_cond.MultipleOrBreak([
    #         break_cond.GenerationBreak(200),
    #         break_cond.MaxFitnessBreak(0, 0.99),
    #         break_cond.MultipleAndBreak([
    #             break_cond.GenerationBreak(100),
    #             break_cond.NotBreak(break_cond.MaxFitnessBreak(0, 0.95))
    #         ])
    #     ]),
    #     "crossover_list": [cross.OnePointElementCrossover(closest_distances)],
    #     "fitness_functions": [
    #         soap_fitness_mid, soap_fitness_weak, soap_fitness_strong 
    #     ],
    #     "mutation_list": all_opti_muts,
    #     **selection_defaults
    # }

    # ╔══════════════════════════════════════════════════════════╗
    # ║                      Initialize Run                      ║
    # ╚══════════════════════════════════════════════════════════╝
    reference_similarity = ff.SimilarityToTargetSOAPFitness(
        target_soap_features=target_features,
        soap_object=soap_obj,
        soap_similarity=soap_sim.RBFSimilarity(
            target_feature_vector=target_features,
            rbf_gamma=0.1,
            adjust_gamma=False,
        ),
        db_title="Reference"
    )

    global_stats_dict = {
        "Reference_Similarity": reference_similarity.evaluate_individual,
        "Volume": lambda x: x.get_volume(),
    }

    multi_stage_search = multi_stage.MultiStageSearch(
        save_dir="data/processed/results/",
        description="Like before, but no multi mutation in optimization.",
        global_statistics_dict=global_stats_dict,
        log_level=log_level
    )

    exploration_defaults = ExlorationGAPreset(
        closest_distances=closest_distances,
        cell_bounds=cell_bounds[0],
        soap_object=soap_obj,
        soap_features=target_features,
    )
    print(exploration_defaults)

    # ╔══════════════════════════════════════════════════════════╗
    # ║                        Run Stages                        ║
    # ╚══════════════════════════════════════════════════════════╝

    from fucrimodo.customs.ga_stage import GAStage

    # Stage 1: Exploration
    population = multi_stage_search.run(
        population=population,
        stage = GAStage(**exploration_defaults())
    )

    exploration_defaults.change_cell_bounds(cell_bounds[1])
    population = multi_stage_search.run(
        population=population,
        stage = GAStage(**exploration_defaults())
    )

    # # Stage 2: Optimization
    # population = multi_stage_search.run(
    #     population=population,
    #     stage = GAStage("Optimization", **optimization_defaults)
    # )
    #
    # # Stage 3: Extended Exploration
    # exploration_defaults["crossover_list"][-1] = cross.StackCellsCrossover(
    #     closest_distances, cell_bounds[1]
    # )
    # exploration_defaults["mutation_list"] = all_muts_2
    # population = multi_stage_search.run(
    #     population=population,
    #     stage = GAStage(f"Extended Exploration", **exploration_defaults)
    # )
    #
    # # Stage 4: Optimization
    # population = multi_stage_search.run(
    #     population=population,
    #     stage = GAStage("Optimization", **optimization_defaults)
    # )

    multi_stage_search.save_results()
