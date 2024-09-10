import ase
from configs.crossover import get_exploration_crossovers
from fucrimodo.core.multi_ga_search import MultiGenAlgSearch
import random
from fucrimodo.customs import population_selections as start_pop
from fucrimodo.core import multi_ga_search as multi_ga
from fucrimodo.core.utils import data_handeling
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.utils.save_current_script import save_current_script
from configs.mutations import get_optimize_mutations, get_all_muts
from ase.io import read as ase_read

import numpy as np
from numpy.typing import NDArray
from icecream import ic
import warnings



def get_stage_list(
    target_soap_features,
    soap_species,
    soap_object,
    exploration_defaults: dict = {
        "number_of_generations": 40,
        "crossover_probability": 0.8,
        "mutation_probability": 0.6,
    },
    optimization_defaults: dict = {
        "number_of_generations": 40,
        "crossover_probability": 0.8,
        "mutation_probability": 0.6,
    },
    additional_statistics_func = None,
    add_stats_func_name = None,
) -> list:

    cell_bounds = []
    for l_max in [4, 6, 8]:
        cell_bounds.append(
            CustomCellBounds({
                "a": [1, l_max], "b": [1, l_max], "c": [1, l_max], 
                "alpha": [20, 160], "beta": [20, 160], "gamma": [20, 160]
            })
        )

    closest_distances = CustomClosestDistances(
        species=soap_species,
        ratio_of_covalent_radii=0.5
    )


    # ── Fitness functions ───────────────────────────────────────────────────
    from configs.fitness_functions import get_soap_similarity_fitness_list, get_species_specific_soap_fitness_list
    species_specific_fitnesses = get_species_specific_soap_fitness_list(
        target_soap_features=target_soap_features,
        soap_species=soap_species,
        soap_object=soap_object,
        rbf_gamma=0.01
    )
    soap_fitness_list = get_soap_similarity_fitness_list(
        target_soap_features=target_soap_features,
        soap_object=soap_object
    )
    soap_fitness_strong = soap_fitness_list[0]
    soap_fitness_mid = soap_fitness_list[1]
    soap_fitness_weak = soap_fitness_list[2]

    population_size = 50


    # ── Break conditions ────────────────────────────────────────────────────
    from configs.break_conditions import exploration_break, optimization_break

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

    # ╒══════════════════════════════════════════════════════════╕
    #                       Define Stages
    # ╘══════════════════════════════════════════════════════════╛

    stage_list = []

    for i in range(len(species_specific_fitnesses)):
        stage_list.append(data_handeling.StageData(
            **exploration_defaults,
            start_population_selection=start_pop.DopePopulationSelection(
                atom_types=soap_species,
                add_n=population_size//len(species_specific_fitnesses),
                cell_bounds=cell_bounds[0],
            ),
            fitness_functions=species_specific_fitnesses[0:i+1],
            crossover_list=explore_cross_1,
            mutation_list=all_muts_1,
            break_condition=exploration_break,
            additional_statistics_func = additional_statistics_func,
            add_stats_func_name = add_stats_func_name
        ))

        stage_list.append(data_handeling.StageData(
            **optimization_defaults,
            start_population_selection=start_pop.SelectAllPopulation(),
            fitness_functions=species_specific_fitnesses[0:i+1] + [soap_fitness_mid],
            crossover_list=explore_cross_1,
            mutation_list=all_muts_1,
            additional_statistics_func=additional_statistics_func,
            add_stats_func_name=add_stats_func_name,
            break_condition=optimization_break
        ))

    stage_list.append(data_handeling.StageData(
        **optimization_defaults,
        start_population_selection=start_pop.SelectAllPopulation(),
        fitness_functions=species_specific_fitnesses,
        crossover_list=explore_cross_1,
        mutation_list=all_opti_muts,
        additional_statistics_func=additional_statistics_func,
        add_stats_func_name=add_stats_func_name,
        break_condition=optimization_break
    ))

    stage_list.append(data_handeling.StageData(
        **exploration_defaults,
        start_population_selection=start_pop.SelectAllPopulation(),
        fitness_functions=[soap_fitness_weak, soap_fitness_mid, (soap_fitness_strong, 0.5)] + species_specific_fitnesses,
        crossover_list=explore_cross_2,
        mutation_list=all_muts_1,
        additional_statistics_func=additional_statistics_func,
        add_stats_func_name=add_stats_func_name,
        break_condition=exploration_break
    ))

    stage_list.append(data_handeling.StageData(
        **optimization_defaults,
        start_population_selection=start_pop.SelectAllPopulation(),
        fitness_functions=[soap_fitness_weak, soap_fitness_mid, (soap_fitness_strong, 0.5)] + species_specific_fitnesses,
        crossover_list=explore_cross_2,
        mutation_list=all_opti_muts,
        additional_statistics_func=additional_statistics_func,
        add_stats_func_name=add_stats_func_name,
        break_condition=optimization_break
    ))

    return stage_list
