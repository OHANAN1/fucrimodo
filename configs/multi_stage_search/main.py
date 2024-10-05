# Description: Main script for running the multi-stage search

from fucrimodo.core.modules.population import Population
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.core.utils.custom_soap import CustomSOAP
import numpy as np

def main(
    target_features: np.ndarray,
    soap_obj: CustomSOAP,
    ):
    import random
    from fucrimodo.core import multi_stage_search as multi_stage
    import numpy as np
    from icecream import ic
    import warnings

    # ╔══════════════════════════════════════════════════════════╗
    # ║                      Debug Settings                      ║
    # ╚══════════════════════════════════════════════════════════╝

    log_enable = False
    warnings.filterwarnings("always")

    random.seed(42)
    np.random.seed(42)

    verbose = 3
    soap_species: list["str"] = soap_obj.species

    # ── Stages ──────────────────────────────────────────────────────────────
    from .stage_list import get_stage_list
    stage_list = get_stage_list(
        soap_object=soap_obj,
        target_soap_features=target_features,
        soap_species=soap_species,
    )

    # ── Global Statistics ───────────────────────────────────────────────────
    from .global_statistics import get_global_statistics_dict
    global_stats_dict = get_global_statistics_dict(
        soap_object=soap_obj,
        target_soap_features=target_features,
    )

    # ── Start Population Candidates ─────────────────────────────────────────
    from .population_generator import get_start_pop_candidates
    population = get_start_pop_candidates(
        soap_species=soap_species,
        population_size=20
    )

    # run_data = data_handeling.RunData(
    #     save_dir="data/processed/results/",
    #     soap_object=self.soap_obj,
    #     log_enable=log_enable,
    #     save_n_best_crystals=10,
    # )
    ic.enable()

    # ── Perform Run ─────────────────────────────────────────────────────────
    # run_data.add_run_settings(
    #     stage_data_list=stage_list,
    #     verbose=verbose
    # )

    multi_stage_search = multi_stage.MultiStageSearch(
        save_dir="data/processed/results/",
        global_statistics_dict=global_stats_dict,
    )


    cell_bounds = CustomCellBounds({
        "a": [1, 4], "b": [1, 4], "c": [1, 4],
        "alpha": [20, 160], "beta": [20, 160], "gamma": [20, 160]
    })
    
    from fucrimodo.customs import population_selections as start_pop
    dope_sel = start_pop.DopePopulationSelection(
        atom_types=soap_species,
        add_n=12,
        cell_bounds=cell_bounds,
    )

    for i in range(2):
        stage = stage_list[i]

        individuals = dope_sel.select(individuals=population.individuals)
        population.individuals = individuals

        population = multi_stage_search.run(
            stage=stage,
            population=population,
        )

    # for i in range(2, len(stage_list)):
    #     stage = stage_list[i]
    #
    #     population = multi_stage_search.run(
    #         stage=stage,
    #         population=population,
    #     )

    # run_data.save_run_info_json()
    multi_stage_search.save_results()

    # save_current_script(run_data)
