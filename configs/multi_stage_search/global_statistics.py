from fucrimodo.core.modules import FitnessFunction
from fucrimodo.core.modules import Individual
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.utils.cellbounds_custom import CustomCellBounds
from fucrimodo.core.utils.custom_soap import CustomSOAP
from fucrimodo.utils import soap_similarity as soap_sim
from collections.abc import Sequence
from fucrimodo.customs import fitness_functions as ff
import ase
from typing import Callable

def get_global_statistics_dict(
    target_soap_features,
    soap_object: CustomSOAP,
) -> dict[str, Callable[[Individual], float]]:
    global_stats_dict = {}

    ref_fitness = ff.SimilarityToTargetSOAPFitness(
        target_soap_features=target_soap_features,
        soap_object=soap_object,
        soap_similarity=soap_sim.RBFSimilarity(
            target_feature_vector=target_soap_features,
            rbf_gamma=0.1,
            adjust_gamma=False,
        )
    )

    global_stats_dict["reference_similarity"] = ref_fitness.evaluate_individual
    global_stats_dict["volume"] = lambda x: x.get_volume()

    return global_stats_dict
