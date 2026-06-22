import numpy as np
from fucrimodo.core.utils import CustomSOAP


def get_present_species(
    soap_obj: CustomSOAP,
    feature_vector: np.ndarray,
    sort_by_appearance: bool = True,
) -> list[str]:
    """Return only those species that contribute to the soap.

    This means all species that have values != 0 in their corresponding
    slice of the whole feature vector.

    :param soap_obj: The soap object that was used to create the
        target features.
    :param target_features: The target soap features.
    :param sort_by_appearance: If set to True, the method tries to guess
        the approximate composition of the target structure by calculating the
        total number of features for each species. The output is then sorted
        by the number of features in descending order.

    :return: A list of species that have features in the target soap.
        If sort_by_appearance is set to True, the list is sorted by the
        number of features in descending order. (The most prominent species
        is at index 0.)
    """
    # Ensure that the analysis is only done once for each species
    unique_soap_obj_species = list(set(soap_obj.species))

    # Loop over all unique species in the soap object
    species_with_features: list[str] = []
    feature_sum_per_species: list[float] = []
    for single_specie in unique_soap_obj_species:
        # Get the slice obj for the part of the feature vector that
        # corresponds to the current species
        species_slice = soap_obj.get_location((single_specie, single_specie))

        # Calculate the sum of absolute values of the feature vector
        feature_vec_abs_sum = float(np.sum(np.abs(feature_vector[species_slice])))

        # If the sum is zero, the species has no features in the provided
        # feature vector
        if feature_vec_abs_sum == 0:
            print(f"Species {single_specie} has no features in descriptor.")
            # continue to the next species so it is not added to the list
            # of species with features
            continue

        else:
            # Add the species to the list of species with features
            species_with_features.append(single_specie)
            feature_sum_per_species.append(feature_vec_abs_sum)

    # If the sort_by_appearance flag is set to True, sort the species
    # by the number of features
    if sort_by_appearance:
        sort_indices = np.argsort(feature_sum_per_species)

        # Reverse the sort order, so that it is from high to low
        sort_indices = sort_indices[::-1]
        species_with_features = [species_with_features[i] for i in sort_indices]

    return species_with_features
