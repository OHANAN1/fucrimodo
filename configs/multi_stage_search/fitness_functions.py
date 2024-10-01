from fucrimodo.core.modules import FitnessFunction
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.core.utils.custom_soap import CustomSOAP
from fucrimodo.utils import soap_similarity as soap_sim
from collections.abc import Sequence
from fucrimodo.customs import fitness_functions as ff

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
    soap_species: list[str],
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
                    species=soap_species[i],
                    species_to_compare=[soap_species[j]],
                ),
                db_title="{}_{}_{}".format(
                    function_name, soap_species[i], soap_species[j]
                )
            )
            species_specific_fitnesses.append(soap_fit_spec)

    return species_specific_fitnesses

