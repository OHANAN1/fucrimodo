from typing import Any
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
from abc import ABC, abstractmethod
from fucrimodo.customs.ga_stage.mutations import Mutation
from fucrimodo.customs.ga_stage.crossovers import Crossover
from fucrimodo.customs.ga_stage.break_conditions import BreakCondition
from fucrimodo.customs.population_selections import PopulationSelection
from fucrimodo.customs.fitness_functions import FitnessFunction

# ╔══════════════════════════════════════════════════════════╗
# ║                    ABC for GA presets                    ║
# ╚══════════════════════════════════════════════════════════╝
# The presets need to be unpacked into the GAStage class initialization with
# e.g. 
# ga_preset=GAPreset 
# GAStage(**ga_preset)

class GAPreset(ABC, dict):
    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        cell_bounds: CustomCellBounds,
        soap_object: CustomSOAP,
        soap_features: np.ndarray,
    ):
        self._closest_distances = closest_distances
        self._cell_bounds = cell_bounds
        self._soap_object = soap_object
        self._soap_features = soap_features

        # Create the dict that is used for the initialization of the GAStage
        self._dict = dict(
            name=self.name,
            fitness_functions=self.fitness_functions,
            crossover_list=self.crossover_list,
            mutation_list=self.mutation_list,
            mutation_probability=self.mutation_probability,
            crossover_probability=self.crossover_probability,
            break_condition=self.break_condition,
            parent_selection=self.parent_selection,
            survivor_selection=self.survivor_selection,
            parent_ratio=self.parent_ratio,
            description=self.description,
            save_n_crystals=self.save_n_crystals
        )

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def fitness_functions(self) -> Sequence[FitnessFunction | tuple[FitnessFunction, float]]:
        pass

    @property
    @abstractmethod
    def crossover_list(self) -> Sequence[Crossover | tuple[Crossover, float]]:
        pass

    @property
    @abstractmethod
    def mutation_list(self) -> Sequence[Mutation | tuple[Mutation, float]]:
        pass

    @property
    @abstractmethod
    def mutation_probability(self) -> float:
        pass

    @property
    @abstractmethod
    def crossover_probability(self) -> float:
        pass

    @property
    @abstractmethod
    def break_condition(self) -> BreakCondition:
        pass

    @property
    @abstractmethod
    def parent_selection(self) -> PopulationSelection:
        pass

    @property
    @abstractmethod
    def survivor_selection(self) -> PopulationSelection:
        pass

    @property
    def parent_ratio(self) -> float:
        return 0.5

    @property
    def description(self) -> str:
        return ""

    @property
    def save_n_crystals(self) -> int:
        return 10

    # Add methods that are required for the dict class to work
    def __call__(self) -> dict[str, Any]:
        return self._dict

    def __getitem__(self, key):
        return self._dict[key]

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def values(self):
        return self._dict.values()

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return f"{self.name} preset"
    

# ╔══════════════════════════════════════════════════════════╗
# ║                     Utility methods                      ║
# ╚══════════════════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════╗
# ║                         Presets                          ║
# ╚══════════════════════════════════════════════════════════╝

class ExlorationGAPreset(GAPreset):
    def change_cell_bounds(self, cell_bounds: CustomCellBounds):
        """Set new cell bounds.

        This resets all properties in the class so they are recalculated with 
        the new cell bounds when called.
        """
        self._cell_bounds = cell_bounds
        self.reset()

    def reset(self):
        """Reset the public properties in the class."""
        del self._fitness_functions
        del self._crossover_list
        del self._mutation_list
        del self._break_condition
        del self._parent_selection
        del self._survivor_selection

    @property
    def name(self) -> str:
        return "Exploration"

    @property
    def fitness_functions(self) -> Sequence[FitnessFunction | tuple[FitnessFunction, float]]:
        if not hasattr(self, "_fitness_functions"):
            soap_fitness_list = get_soap_similarity_fitness_list(
                target_soap_features=self._soap_features,
                soap_object=self._soap_object
            )
            species_specific_fitnesses = get_species_specific_soap_fitness_list(
                target_soap_features=self._soap_features,
                soap_species=self._soap_object.species,
                soap_object=self._soap_object,
                rbf_gamma=0.01
            )
            self._fitness_functions = soap_fitness_list + species_specific_fitnesses

        return self._fitness_functions

    @property
    def crossover_list(self) -> Sequence[Crossover | tuple[Crossover, float]]:
        if not hasattr(self, "_crossover_list"):
            self._crossover_list =[
                cross.OnePointElementCrossover(self._closest_distances),
                cross.UnitCellCrossover(self._closest_distances),
            ]
        return self._crossover_list
    
    @property
    def mutation_list(self) -> Sequence[Mutation | tuple[Mutation, float]]:
        if not hasattr(self, "_mutation_list"):
            all_muts = [
                mut.elem_mut.PermutationMutation(
                    closest_distances=self._closest_distances,
                ),
                mut.pos_mut.RattleMutation(
                    closest_distances=self._closest_distances, n_top=1, rattle_strength=0.1
                ),
                mut.elem_mut.AddAtomsMutation(
                    possible_elements=self._soap_object.species,
                    closest_distances=self._closest_distances
                ),
                mut.elem_mut.DeleteRandomAtomsMutation(
                    closest_distances=self._closest_distances
                ),
                mut.elem_mut.ReplaceAtomsMutation(
                    possible_elements=self._soap_object.species,
                    closest_distances=self._closest_distances
                ),
                mut.energy_mut.SoftMutation(
                    closest_distances=self._closest_distances
                ),
                mut.cell_mut.MinimizeTiltMutation(
                    closest_distances=self._closest_distances
                ),
                mut.sym_mut.GetConventionalCellMutation(
                    closest_distances=self._closest_distances,
                    symmetry_tol=0.3
                ),
                mut.cell_mut.RotationMutation(
                    closest_distances=self._closest_distances
                ),
                mut.cell_mut.EnlargeMutation(
                    closest_distances=self._closest_distances,
                    cell_bounds=self._cell_bounds
                ),
                mut.cell_mut.StrainMutation(
                    closest_distances=self._closest_distances,
                    n_variable_cell_vectors=3,
                    cell_bounds=self._cell_bounds,
                ),
                mut.cell_mut.CutoutMutation(
                    closest_distances=self._closest_distances,
                    cell_bounds=self._cell_bounds,
                    tolerance=1.
                )
            ]
            multi_mut = mut.multi_mut.MultipleMutations(
                mutations=all_muts,
                number_of_mutations=2,
                random_order=True,
                closest_distances=self._closest_distances
            )
            self._mutation_list = all_muts + [multi_mut]
        return self._mutation_list

    @property
    def mutation_probability(self) -> float:
        if not hasattr(self, "_mutation_probability"):
            self._mutation_probability = 0.8
        return self._mutation_probability

    @property
    def crossover_probability(self) -> float:
        if not hasattr(self, "_crossover_probability"):
            self._crossover_probability = 0.8
        return self._crossover_probability

    @property
    def break_condition(self) -> BreakCondition:
        if not hasattr(self, "_break_condition"):
            self._break_condition = break_cond.MultipleOrBreak([
                break_cond.GenerationBreak(5),
                break_cond.MaxFitnessBreak(0, 0.85),
                break_cond.MultipleAndBreak([
                    break_cond.GenerationBreak(100),
                    break_cond.NotBreak(break_cond.MaxFitnessBreak(0, 0.80))
                ])
            ])
        return self._break_condition

    @property
    def parent_selection(self) -> PopulationSelection:
        if not hasattr(self, "_parent_selection"):
            self._parent_selection = pop_sel.TournamentSelection(tournament_size=4)
        return self._parent_selection

    @property
    def survivor_selection(self) -> PopulationSelection:
        if not hasattr(self, "_survivor_selection"):
            self._survivor_selection = pop_sel.NSGA2Selection()
        return self._survivor_selection

    @property
    def description(self) -> str:
        return "Apply strong modifications to the population to explore the search space. (Preset: Exploration)"

    @property
    def save_n_crystals(self) -> int:
        return 10
