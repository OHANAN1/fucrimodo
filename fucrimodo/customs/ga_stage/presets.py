import numpy as np
from fucrimodo.customs import population_selections as pop_sel
from fucrimodo.customs.ga_stage import break_conditions as break_cond
import numpy as np
from fucrimodo.customs import fitness_functions as ff
from abc import ABC
from collections.abc import Sequence
from fucrimodo.customs.ga_stage import mutations as mut
from fucrimodo.customs.ga_stage import crossovers as cross
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.utils.custom_soap import CustomSOAP
from fucrimodo.core.modules import FitnessFunction
from fucrimodo.customs.ga_stage.ga_stage import GAStage
from fucrimodo.utils import soap_similarity as soap_sim
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

class GAPreset(ABC):
    """Abstract base class for GAStage
    This class is used to create presets for the GAStage class.
    The properties of the class are used to set the properties of the GAStage
    object.
    The properties are set when the GAStage object is created.
    Use `del` to reset the properties of the class to its default values.
    """
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

    @property
    def name(self) -> str:
        if not hasattr(self, "_name"):
            self._name = "No name set, please set name attribute."
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = str(value)

    @property
    def fitness_functions(self) -> Sequence[FitnessFunction | tuple[FitnessFunction, float]]:
        # Prompt the user to implement the property
        # Normally this would be done with an abstract property,
        # but this is not possible with the setter
        raise NotImplementedError(
            "Please implement the fitness_functions property."
        )

    @fitness_functions.setter
    def fitness_functions(self, value: Sequence[FitnessFunction | tuple[FitnessFunction, float]]):
        self._fitness_functions = value

    @property
    def crossover_list(self) -> Sequence[cross.Crossover | tuple[cross.Crossover, float]]:
        # Prompt the user to implement the property
        # Normally this would be done with an abstract property, 
        # but this is not possible with the setter
        raise NotImplementedError(
            "Please implement the crossover_list property."
        )

    @crossover_list.setter
    def crossover_list(self, value: Sequence[cross.Crossover | tuple[cross.Crossover, float]]):
        self._crossover_list = value

    @property
    def mutation_list(self) -> Sequence[mut.Mutation | tuple[mut.Mutation, float]]:
        # Prompt the user to implement the property
        # Normally this would be done with an abstract property,
        # but this is not possible with the setter
        raise NotImplementedError(
            "Please implement the mutation_list property."
        )

    @mutation_list.setter
    def mutation_list(self, value: Sequence[mut.Mutation | tuple[mut.Mutation, float]]):
        self._mutation_list = value

    @property
    def mutation_probability(self) -> float:
        if not hasattr(self, "_mutation_probability"):
            self._mutation_probability = 0.8
        return self._mutation_probability

    @mutation_probability.setter
    def mutation_probability(self, value: float):
        self._mutation_probability = value

    @property
    def crossover_probability(self) -> float:
        if not hasattr(self, "_crossover_probability"):
            self._crossover_probability = 0.8
        return self._crossover_probability

    @crossover_probability.setter
    def crossover_probability(self, value: float):
        self._crossover_probability = value

    @property
    def break_condition(self) -> BreakCondition:
        if not hasattr(self, "_break_condition"):
            self._break_condition = break_cond.NeverBreak()
        return self._break_condition

    @break_condition.setter
    def break_condition(self, value: BreakCondition):
        self._break_condition = value

    @property
    def parent_selection(self) -> PopulationSelection:
        if not hasattr(self, "_parent_selection"):
            self._parent_selection = pop_sel.TournamentSelection(
                tournament_size=4
            )
        return self._parent_selection

    @parent_selection.setter
    def parent_selection(self, value: PopulationSelection):
        self._parent_selection = value

    @property
    def survivor_selection(self) -> PopulationSelection:
        if not hasattr(self, "_survivor_selection"):
            self._survivor_selection = pop_sel.NSGA2Selection()
        return self._survivor_selection

    @survivor_selection.setter
    def survivor_selection(self, value: PopulationSelection):
        self._survivor_selection = value

    @property
    def parent_ratio(self) -> float:
        if not hasattr(self, "_parent_ratio"):
            self._parent_ratio = 0.5
        return self._parent_ratio

    @parent_ratio.setter
    def parent_ratio(self, value: float):
        self._parent_ratio = value

    @property
    def description(self) -> str:
        if not hasattr(self, "_description"):
            self._description = ""
        return self._description

    @description.setter
    def description(self, value: str):
        self._description = value

    @property
    def save_n_crystals(self) -> int:
        if not hasattr(self, "_save_n_crystals"):
            self._save_n_crystals = 10
        return self._save_n_crystals

    @save_n_crystals.setter
    def save_n_crystals(self, value: int):
        self._save_n_crystals = value

    def create(self) -> GAStage:
        """Create the GAStage object with the preset properties."""
        return GAStage(
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

    def change_cell_bounds(self, cell_bounds: CustomCellBounds):
        """Set new cell bounds.

        This resets the mutations and the crossovers so they are recalculated
        with the new cell bounds when called.
        """
        self._cell_bounds = cell_bounds

        # Reset the mutations and crossovers if they are already set as
        # private properties so they are recalculated with the new cell bounds.
        if hasattr(self, "_mutation_list"):
            del self._mutation_list
        if hasattr(self, "_crossover_list"):
            del self._crossover_list

    def reset(self):
        """Reset the properties in the class that are used to init the GAStage.
        Appart from the name and description.

        The properties are recalculated and set when called.
        Usefull, when the cell bounds or the closest distances are changed, 
        so the properties are recalculated with the new values.
        """
        if hasattr(self, "_fitness_functions"):
            del self._fitness_functions

        if hasattr(self, "_mutation_list"):
            del self._mutation_list

        if hasattr(self, "_crossover_list"):
            del self._crossover_list

        if hasattr(self, "_break_condition"):
            del self._break_condition
            
        if hasattr(self, "_parent_selection"):
            del self._parent_selection

        if hasattr(self, "_survivor_selection"):
            del self._survivor_selection


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
    # Only adjust the getters, so the properties of the GAPreset are not 
    # overwritten completely.
    @GAPreset.name.getter
    def name(self) -> str:
        if not hasattr(self, "_name"):
            self._name = "Exploration GA"
        return self._name

    @GAPreset.description.getter
    def description(self) -> str:
        return "Apply strong modifications to the population to explore the search space. (Preset: Exploration)"

    @GAPreset.fitness_functions.getter
    def fitness_functions(self) -> Sequence[FitnessFunction | tuple[FitnessFunction, float]]:
        if not hasattr(self, "_fitness_functions"):
            # Order the fitness functions in a way that the reference similarity
            # (gamma = 0.1) is the first fitness function.
            # The break conditions react to the first fitness function.
            soap_fitness_list = get_soap_similarity_fitness_list(
                target_soap_features=self._soap_features,
                soap_object=self._soap_object,
                rbf_gammas=[0.1, 1.0, 0.01],
                function_titles=[
                    "soap_similarity_mid",
                    "soap_similarity_strong",
                    "soap_similarity_weak"
                ]
            )
            species_specific_fitnesses = get_species_specific_soap_fitness_list(
                target_soap_features=self._soap_features,
                soap_species=self._soap_object.species,
                soap_object=self._soap_object,
                rbf_gamma=0.01
            )
            self._fitness_functions = soap_fitness_list + species_specific_fitnesses

        return self._fitness_functions

    @GAPreset.crossover_list.getter
    def crossover_list(self) -> Sequence[cross.Crossover | tuple[cross.Crossover, float]]:
        if not hasattr(self, "_crossover_list"):
            self._crossover_list =[
                cross.OnePointElementCrossover(self._closest_distances),
                cross.OnePointPositionCrossover(self._closest_distances),
                cross.UnitCellCrossover(self._closest_distances),
            ]
        return self._crossover_list
    
    @GAPreset.mutation_list.getter
    def mutation_list(self) -> Sequence[mut.Mutation | tuple[mut.Mutation, float]]:
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

    @GAPreset.break_condition.getter
    def break_condition(self) -> BreakCondition:
        if not hasattr(self, "_break_condition"):
            self._break_condition = break_cond.MultipleOrBreak([
                break_cond.GenerationBreak(200),
                break_cond.MaxFitnessBreak(0, 0.85),
                break_cond.MultipleAndBreak([
                    break_cond.GenerationBreak(100),
                    break_cond.NotBreak(break_cond.MaxFitnessBreak(0, 0.80))
                ])
            ])
        return self._break_condition


class OptimizationGAPreset(GAPreset):
    # Only adjust the getters, so the properties of the GAPreset are not 
    # overwritten completely.
    @GAPreset.name.getter
    def name(self) -> str:
        if not hasattr(self, "_name"):
            self._name = "Optimization GA"
        return self._name

    @GAPreset.description.getter
    def description(self) -> str:
        return (
            "Apply weak modifications to the population that do not create "
                "entirely new individuals but slightly modify the ones "
                "existing. (Preset: Optimization)"
        )

    @GAPreset.fitness_functions.getter
    def fitness_functions(self) -> Sequence[FitnessFunction | tuple[FitnessFunction, float]]:
        if not hasattr(self, "_fitness_functions"):
            # Order the fitness functions in a way that the reference similarity
            # (gamma = 0.1) is the first fitness function.
            # The break conditions react to the first fitness function.
            self._fitness_functions = get_soap_similarity_fitness_list(
                target_soap_features=self._soap_features,
                soap_object=self._soap_object,
                rbf_gammas=[0.1, 1.0, 0.01],
                function_titles=[
                    "soap_similarity_mid",
                    "soap_similarity_strong",
                    "soap_similarity_weak"
                ]
            )
        return self._fitness_functions

    @GAPreset.crossover_list.getter
    def crossover_list(self) -> Sequence[cross.Crossover | tuple[cross.Crossover, float]]:
        if not hasattr(self, "_crossover_list"):
            self._crossover_list =[
                cross.OnePointElementCrossover(self._closest_distances),
                cross.OnePointPositionCrossover(self._closest_distances),
            ]
        return self._crossover_list
    
    @GAPreset.mutation_list.getter
    def mutation_list(self) -> Sequence[mut.Mutation | tuple[mut.Mutation, float]]:
        if not hasattr(self, "_mutation_list"):
            all_muts = [
                mut.elem_mut.PermutationMutation(
                    closest_distances=self._closest_distances,
                ),
                mut.pos_mut.RattleMutation(
                    closest_distances=self._closest_distances, 
                    n_top=1, 
                    rattle_strength=0.1
                ),
                mut.pos_mut.RattleMutation(
                    closest_distances=self._closest_distances, 
                    n_top=1, 
                    rattle_strength=0.5
                ),
                mut.pos_mut.RattleMutation(
                    closest_distances=self._closest_distances, 
                    n_top=1,
                    rattle_strength=0.01
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
            ]
            multi_mut = mut.multi_mut.MultipleMutations(
                mutations=all_muts,
                number_of_mutations=2,
                random_order=True,
                closest_distances=self._closest_distances
            )
            self._mutation_list = all_muts + [multi_mut]
        return self._mutation_list

    @GAPreset.break_condition.getter
    def break_condition(self) -> BreakCondition:
        if not hasattr(self, "_break_condition"):
            self._break_condition = break_cond.MultipleOrBreak([
                break_cond.GenerationBreak(200),
                break_cond.MaxFitnessBreak(0, 0.99),
                break_cond.MultipleAndBreak([
                    break_cond.GenerationBreak(100),
                    break_cond.NotBreak(break_cond.MaxFitnessBreak(0, 0.80))
                ])
            ])
        return self._break_condition

