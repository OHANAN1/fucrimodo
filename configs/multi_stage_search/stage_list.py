from .crossover import get_exploration_crossovers
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from .mutations import get_optimize_mutations, get_all_muts


def get_stage_list(
    target_soap_features,
    soap_species,
    soap_object,
    exploration_defaults: dict = {
        "crossover_probability": 0.8,
        "mutation_probability": 0.6,
    },
    optimization_defaults: dict = {
        "crossover_probability": 0.8,
        "mutation_probability": 0.6,
    }
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
    from .fitness_functions import get_soap_similarity_fitness_list, get_species_specific_soap_fitness_list
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
    from .break_conditions import exploration_break, optimization_break

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
    from fucrimodo.customs.ga_stage import GAStage

    stage_list = []

    for i in range(len(species_specific_fitnesses)):
        stage_list.append(GAStage(
            name=f"Build up Exploration {i+1}",
            **exploration_defaults,
            fitness_functions=species_specific_fitnesses[0:i+1],
            crossover_list=explore_cross_1,
            mutation_list=all_muts_1,
            break_condition=exploration_break,
        ))

        stage_list.append(GAStage(
            name=f"Build up Optimization {i+1}",
            **optimization_defaults,
            fitness_functions=species_specific_fitnesses[0:i+1] + [soap_fitness_mid],
            crossover_list=explore_cross_1,
            mutation_list=all_muts_1,
            break_condition=optimization_break
        ))

    stage_list.append(GAStage(
        name="Exploration",
        **optimization_defaults,
        fitness_functions=species_specific_fitnesses,
        crossover_list=explore_cross_1,
        mutation_list=all_opti_muts,
        break_condition=optimization_break
    ))

    stage_list.append(GAStage(
        name="Exploration",
        **exploration_defaults,
        fitness_functions=[soap_fitness_weak, soap_fitness_mid, (soap_fitness_strong, 0.5)] + species_specific_fitnesses,
        crossover_list=explore_cross_2,
        mutation_list=all_muts_1,
        break_condition=exploration_break
    ))

    stage_list.append(GAStage(
        name="Exploration",
        **optimization_defaults,
        fitness_functions=[soap_fitness_weak, soap_fitness_mid, (soap_fitness_strong, 0.5)] + species_specific_fitnesses,
        crossover_list=explore_cross_2,
        mutation_list=all_opti_muts,
        break_condition=optimization_break
    ))

    return stage_list
