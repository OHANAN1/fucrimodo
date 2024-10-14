# Description: Main script for running the multi-stage search
import numpy as np
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.utils.custom_soap import CustomSOAP
import random
from fucrimodo.core import multi_stage_search as multi_stage
import numpy as np
import logging
from fucrimodo.utils import soap_similarity as soap_sim
from fucrimodo.customs import fitness_functions as ff
from fucrimodo.core.modules import Individual
from fucrimodo.core.modules import Population
from fucrimodo.customs import population_generator as crystal_creation
from fucrimodo.customs.ga_stage.presets import ExlorationGAPreset, OptimizationGAPreset
from fucrimodo.customs.ga_stage import GAStage

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
    verbose: int = 3,
    random_seed: int = 42,
    ):
    # ── Set random seed ─────────────────────────────────────────────────────
    random.seed(random_seed)
    np.random.seed(random_seed)

    # ── Global Setup ───────────────────────────────────────────────────
    soap_species = soap_obj.species

    closest_distances = CustomClosestDistances(
        species=soap_species,
        ratio_of_covalent_radii=0.8
    )

    cell_bounds = []
    for l_max in [8, 14]:
        cell_bounds.append(
            CustomCellBounds({
                "a": [1, l_max], "b": [1, l_max], "c": [1, l_max],
                "alpha": [20, 160], "beta": [20, 160], "gamma": [20, 160],
                "phi": [0, 180], "chi": [0, 180], "psi": [0, 180]
            })
        )


    # ── Setup Global Statistics ─────────────────────────────────────────────
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


    # ── Start Population Candidates ─────────────────────────────────────────
    population = get_start_pop_candidates(
        soap_species=soap_species,
        population_size=100
    )

    # ╔══════════════════════════════════════════════════════════╗
    # ║                      Load Defaults                       ║
    # ╚══════════════════════════════════════════════════════════╝

    explore_ga = ExlorationGAPreset(
        closest_distances=closest_distances,
        cell_bounds=cell_bounds[0],
        soap_object=soap_obj,
        soap_features=target_features,
    )

    optimize_ga = OptimizationGAPreset(
        closest_distances=closest_distances,
        cell_bounds=cell_bounds[0],
        soap_object=soap_obj,
        soap_features=target_features,
    )

    # ╔══════════════════════════════════════════════════════════╗
    # ║                        Run Stages                        ║
    # ╚══════════════════════════════════════════════════════════╝

    # Setup multi-stage search object
    multi_stage_search = multi_stage.MultiStageSearch(
        save_dir="data/processed/results/",
        description="Like before, but no multi mutation in optimization.",
        global_statistics_dict=global_stats_dict,
        log_level=log_level
    )

    # Stage 1
    population = multi_stage_search.run(
        population=population,
        stage = explore_ga.create()
    )

    # Stage 2
    population = multi_stage_search.run(
        population=population,
        stage = optimize_ga.create()
    )

    # Stage 3
    optimize_ga.name = "Optimization 2"
    explore_ga.change_cell_bounds(cell_bounds[1])
    population = multi_stage_search.run(
        population=population,
        stage = explore_ga.create()
    )

    # Stage 4
    optimize_ga.crossover_probability = 0.9
    population = multi_stage_search.run(
        population=population,
        stage = optimize_ga.create()
    )

    multi_stage_search.save_results()
