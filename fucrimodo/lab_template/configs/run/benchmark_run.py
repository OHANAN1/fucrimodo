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
from fucrimodo.customs import fitness_functions as ff, population_selections
from fucrimodo.core.modules import Individual
from fucrimodo.core.modules import Population
from fucrimodo.customs import population_generator as crystal_creation
from fucrimodo.customs.ga_stage.presets import ExlorationGAPreset, OptimizationGAPreset
from fucrimodo.customs.ga_stage import GAStage, break_conditions


def setup(multi_stage_search: multi_stage.MultiStageSearch):

    # ── Set random seed ─────────────────────────────────────────────────────
    random.seed(42)
    np.random.seed(42)

    # ── Global Setup ───────────────────────────────────────────────────
    soap_species = multi_stage_search.descriptor_object.species

    closest_distances = CustomClosestDistances(
        species=soap_species,
        ratio_of_covalent_radii=0.7
    )

    cell_bounds = [
        CustomCellBounds({
            "a": [1, 8], "b": [1, 8], "c": [1, 8],
            "alpha": [20, 160], "beta": [20, 160], "gamma": [20, 160],
        }),
        CustomCellBounds({
            "a": [1, 14], "b": [1, 14], "c": [1, 14],
            "alpha": [20, 160], "beta": [20, 160], "gamma": [20, 160],
        }),
    ]

    # ── Setup Population Generator ─────────────────────────────────────────
    pop_generator = crystal_creation.OneAtomicCrystalGenerator(
        atom_types=soap_species,
        cell_bounds=cell_bounds[0],
        closest_distances=closest_distances,
        volume=100
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
        db_title="Reference"
    )

    multi_stage_search.global_statistics_dict = {
        "Reference_Similarity": reference_similarity.evaluate_individual,
        "Volume": lambda x: x.get_volume(),
    }

    return closest_distances, cell_bounds, pop_generator


def main(multi_stage_search: multi_stage.MultiStageSearch):
    multi_stage_search.description = "The benchmark run for the multi-stage search."

    # Run the setup
    closest_distances, cell_bounds, pop_generator = setup(multi_stage_search)

    # ╔══════════════════════════════════════════════════════════╗
    # ║                      Load Presets                        ║
    # ╚══════════════════════════════════════════════════════════╝

    explore_ga = ExlorationGAPreset(
        closest_distances=closest_distances,
        cell_bounds=cell_bounds[0],
        soap_object=multi_stage_search.descriptor_object,
        soap_features=multi_stage_search.target_features,
    )

    optimize_ga = OptimizationGAPreset(
        closest_distances=closest_distances,
        cell_bounds=cell_bounds[0],
        soap_object=multi_stage_search.descriptor_object,
        soap_features=multi_stage_search.target_features,
    )


    # ╔══════════════════════════════════════════════════════════╗
    # ║                        Run Stages                        ║
    # ╚══════════════════════════════════════════════════════════╝

    # ── Start Population Candidates ─────────────────────────────────────────
    population = pop_generator.generate_population(10)


    # ── Stage 1 ─────────────────────────────────────────────────────────────
    # Run the exploration GA with the initial population
    population = multi_stage_search.run(
        population=population,
        stage=explore_ga.create()
    )


    # ── Stage 2 ─────────────────────────────────────────────────────────────
    # Run the optimization GA with the population from the exploration
    population = multi_stage_search.run(
        population=population,
        stage=optimize_ga.create()
    )


    # ── Stage 3 ─────────────────────────────────────────────────────────────
    explore_ga.name = "Extended Exploration"
    explore_ga.description = "Run an exploration with bigger cell bounds."

    # Increase the cell bounds to explore bigger structures
    explore_ga.change_cell_bounds(cell_bounds[1])

    # Run the extended exploration
    population = multi_stage_search.run(
        population=population,
        stage=explore_ga.create()
    )


    # ── Stage 4 ─────────────────────────────────────────────────────────────
    optimize_ga.name = "Final Optimization"
    optimize_ga.description = "Run long optimization on the best 10 individuals."

    # Reduce the population to the best 10 individuals
    population.individuals = population_selections.NSGA2Selection().select(
        population.individuals, 10
    )

    # Set the break condition to run way longer
    optimize_ga.break_condition = break_conditions.GenerationBreak(500)

    # Run the final optimization
    population = multi_stage_search.run(
        population=population,
        stage=optimize_ga.create()
    )
