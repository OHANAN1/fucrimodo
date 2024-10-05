# Description: Main script for running the multi-stage search

from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.core.utils.custom_soap import CustomSOAP
import numpy as np

from .crossover import get_exploration_crossovers
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from .mutations import get_optimize_mutations, get_all_muts
from fucrimodo.customs import population_selections as pop_sel

import random
from fucrimodo.core import multi_stage_search as multi_stage
from fucrimodo.customs.ga_stage import break_conditions as break_cond
import numpy as np
from icecream import ic
import warnings

def main(
    target_features: np.ndarray,
    soap_obj: CustomSOAP,
    log_enable: bool = False,
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

    from .global_statistics import get_global_statistics_dict
    global_stats_dict = get_global_statistics_dict(
        soap_object=soap_obj,
        target_soap_features=target_features,
    )

    multi_stage_search = multi_stage.MultiStageSearch(
        save_dir="data/processed/results/",
        global_statistics_dict=global_stats_dict,
    )

    cell_bounds = []
    for l_max in [4, 6, 8]:
        cell_bounds.append(
            CustomCellBounds({
                "a": [1, l_max], "b": [1, l_max], "c": [1, l_max], 
                "alpha": [20, 160], "beta": [20, 160], "gamma": [20, 160]
            })
        )

    dope_sel = pop_sel.DopePopulationSelection(
        atom_types=soap_species,
        cell_bounds=cell_bounds[0],
    )

    # ── Start Population Candidates ─────────────────────────────────────────
    from .population_generator import get_start_pop_candidates
    population = get_start_pop_candidates(
        soap_species=soap_species,
        population_size=20
    )

    closest_distances = CustomClosestDistances(
        species=soap_species,
        ratio_of_covalent_radii=0.5
    )


    # ── Fitness functions ───────────────────────────────────────────────────
    from .fitness_functions import get_soap_similarity_fitness_list, get_species_specific_soap_fitness_list
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
    all_opti_muts = get_optimize_mutations(closest_distances=closest_distances)
    all_muts_1 = get_all_muts(
        closest_distances=closest_distances,
        cell_bounds=cell_bounds[1],
        soap_species=soap_species
    )


    # ── Crossovers ──────────────────────────────────────────────────────────
    explore_cross_1 = get_exploration_crossovers(
        closest_distances=closest_distances,
        cell_bounds=cell_bounds[1]
    )
    explore_cross_2 = get_exploration_crossovers(
        closest_distances=closest_distances,
        cell_bounds=cell_bounds[2]
    )

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
        "mutation_probability": 0.6,
        "break_condition": break_cond.GenerationBreak(10)
    }
    optimization_defaults = {
        "crossover_probability": 0.8,
        "mutation_probability": 0.6,
        "break_condition": break_cond.GenerationBreak(10)
    }


    # ╒══════════════════════════════════════════════════════════╕
    #                       Define Stages
    # ╘══════════════════════════════════════════════════════════╛
    from fucrimodo.customs.ga_stage import GAStage

    n_build_up = len(species_specific_fitnesses)
    for i in range(n_build_up):
        population = multi_stage_search.run(
            population=population,
            stage = GAStage(
                name=f"Build up Exploration {i+1}",
                **exploration_defaults,
                **selection_defaults,
                fitness_functions=species_specific_fitnesses[0:i+1],
                crossover_list=explore_cross_1,
                mutation_list=all_muts_1,
            )
        )

        population = multi_stage_search.run(
            population=population,
            stage = GAStage(
                name=f"Build up Optimization {i+1}",
                **optimization_defaults,
                **selection_defaults,
                fitness_functions=species_specific_fitnesses[0:i+1] + [soap_fitness_mid],
                crossover_list=explore_cross_1,
                mutation_list=all_muts_1,
            )
        )

        # Dope the population with new random individuals
        population.individuals = dope_sel.select(population.individuals, n=12)

    population = multi_stage_search.run(
        population=population,
        stage = GAStage(
            name="Optimization",
            **optimization_defaults,
            **selection_defaults,
            fitness_functions=species_specific_fitnesses,
            crossover_list=explore_cross_1,
            mutation_list=all_opti_muts,
        )
    )

    multi_stage_search.save_results()
