import ase
from fucrimodo.core.multi_ga_search import MultiGenAlgSearch
import random
from fucrimodo.customs import population_selections as start_pop
from fucrimodo.customs import crossovers as cross
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


# ╔══════════════════════════════════════════════════════════╗
# ║                      Debug Settings                      ║
# ╚══════════════════════════════════════════════════════════╝

log_enable = True
warnings.filterwarnings("ignore")
ic.disable()

random.seed(42)
np.random.seed(42)

def main(
    run_data: data_handeling.RunData,
    population_size: int,
    target_soap_features: NDArray[np.float64],
    closest_distances: CustomClosestDistances,
    cell_bounds: list[CustomCellBounds],
    secrets: dict = {}
):
    verbose = 3
    soap_species: list["str"] = run_data.soap_object.species  # type: ignore

    # ── Fitness functions ───────────────────────────────────────────────────
    from configs.fitness_functions import get_soap_similarity_fitness_list, get_species_specific_soap_fitness_list
    species_specific_fitnesses = get_species_specific_soap_fitness_list(
        target_soap_features=target_soap_features,
        soap_species=soap_species,
        soap_object=run_data.soap_object,
        rbf_gamma=0.01
    )
    soap_fitness_list = get_soap_similarity_fitness_list(
        target_soap_features=target_soap_features,
        soap_object=run_data.soap_object
    )
    soap_fitness_strong = soap_fitness_list[0]
    soap_fitness_mid = soap_fitness_list[1]
    soap_fitness_weak = soap_fitness_list[2]


    # ── Start Population Candidates ─────────────────────────────────────────
    from configs.population_generator import get_start_pop_candidates
    start_pop_candidates = get_start_pop_candidates(
        soap_species=soap_species,
        cell_bounds=cell_bounds[0],
        population_size=population_size
    )


    # ── Break conditions ────────────────────────────────────────────────────
    from configs.break_conditions import exploration_break, optimization_break
    n_gens_exploration = 20
    n_gen_optimization = 40


    # ── Mutations ───────────────────────────────────────────────────────────
    all_opti_muts = get_optimize_mutations(closest_distances=closest_distances)
    all_muts_1 = get_all_muts(
        closest_distances=closest_distances,
        cell_bounds=cell_bounds[1],
        soap_species=soap_species
    )
    all_muts_2 = get_all_muts(
        closest_distances=closest_distances,
        cell_bounds=cell_bounds[2],
        soap_species=soap_species
    )

    # Stages
    from configs.stage_list import get_stage_list
    stage_list = get_stage_list(
        soap_fitness_list, 
        species_specific_fitnesses,
        soap_species,
        cell_bounds,
        closest_distances,
        population_size,
        exploration_break,
        optimization_break,
        all_muts_1,
        all_opti_muts,
        additional_statistics_func=soap_fitness_mid.evaluate_individual,
        add_stats_func_name="statistic_similarity",
    )

    run_data.add_run_settings(
        stage_data_list=stage_list,
        verbose=verbose
    )

    ga_grid_search = multi_ga.MultiGenAlgSearch(
        run_data=run_data,
    )
    ga_grid_search.run(
        start_pop_candidates=start_pop_candidates,
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Test of parsers.')
    parser.add_argument(
        '-t', '--target_crystal', type=str,
        help='Give the path to the target SOAP descriptor (file-type: all accepted by ase.io.read).'
    )

    args = parser.parse_args()

    if args.target_crystal is None:
        print("Please define path to target cystal with flag -t!")
        sys.exit()


    target_crystal = ase_read(args.target_crystal)
    if isinstance(target_crystal, ase.Atoms):
        target_atom_numbers = target_crystal.get_atomic_numbers()
    else:
        print("Given path to target could not be parsed to valid ase.Atoms object")
        sys.exit()

    POPULATION_SIZE = 50
    cell_bounds = []
    for l_max in [4, 6, 8]:
        cell_bounds.append(
            CustomCellBounds({
                "a": [1, l_max], "b": [1, l_max], "c": [1, l_max], 
                "alpha": [20, 160], "beta": [20, 160], "gamma": [20, 160]
            })
        )

    closest_distances = CustomClosestDistances(
        species=target_atom_numbers,
        ratio_of_covalent_radii=0.5
    )

    run_data = data_handeling.RunData(
        save_dir="data/processed/results/",
        soap_params={
            "species": np.unique(
                target_crystal.get_chemical_symbols()).tolist(),
            "r_cut": 15.0,
            "n_max": 8,
            "l_max": 8,
            "sigma": 0.5,
        },
        log_enable=log_enable,
        save_n_best_crystals=10,
    )
    run_data.add_crystal_to_database(target_crystal, {"is_target": True})
    save_current_script(run_data)

    main(
        run_data=run_data,
        population_size=POPULATION_SIZE,
        target_soap_features=run_data.soap_object.create(target_crystal),
        closest_distances=closest_distances,
        cell_bounds=cell_bounds,
        secrets={"target_crystal": target_crystal}
    )
