import numpy as np
import ase
from fucrimodo.customs.utils import (
    get_soap_similarity_fitness_list,
    get_species_specific_soap_sim_fitness_list,
)
from fucrimodo.core import Individual
from fucrimodo.customs.fitness_functions import (
    SoapRbfSimilarityFitness,
    SpeciesSpecificSoapRbfSimFitness,
)


def test_get_soap_similarity_fitness_list(periodic_soap_obj):
    fitness_list = get_soap_similarity_fitness_list(
        target_soap_features=np.array([1, 2, 3]),
        soap_object=periodic_soap_obj,
        rbf_gammas=[0.1],
        function_titles=["test"],
        round_result=6,
        n_jobs=5,
    )

    assert len(fitness_list) == 1
    assert isinstance(fitness_list[0], SoapRbfSimilarityFitness)
    assert fitness_list[0].db_title == "test"
    assert fitness_list[0].n_jobs == 5
    assert fitness_list[0].round_result == 6
    assert np.allclose(fitness_list[0].target_soap_features, np.array([1, 2, 3]))

    fitness_list = get_soap_similarity_fitness_list(
        target_soap_features=np.array([1, 2, 3]),
        soap_object=periodic_soap_obj,
        rbf_gammas=[0.1, 1.0],
        function_titles=["test", "other"],
    )
    assert len(fitness_list) == 2
    assert fitness_list[0].db_title == "test"
    assert fitness_list[1].db_title == "other"

    # Test defaults
    fitness_list = get_soap_similarity_fitness_list(
        target_soap_features=np.array([1, 2, 3]),
        soap_object=periodic_soap_obj,
    )
    assert len(fitness_list) == 3
    assert fitness_list[0].db_title == "soap_similarity_strong"
    assert fitness_list[1].db_title == "soap_similarity_mid"
    assert fitness_list[2].db_title == "soap_similarity_weak"

    assert isinstance(fitness_list[0], SoapRbfSimilarityFitness)
    assert isinstance(fitness_list[1], SoapRbfSimilarityFitness)
    assert isinstance(fitness_list[2], SoapRbfSimilarityFitness)

    assert fitness_list[0].rbf_gamma == 1.0
    assert fitness_list[1].rbf_gamma == 0.1
    assert fitness_list[2].rbf_gamma == 0.01


def test_get_species_specific_soap_sim_fitness_list(periodic_soap_obj):
    fitness_list = get_species_specific_soap_sim_fitness_list(
        target_soap_features=np.array([1, 2, 3]),
        soap_object=periodic_soap_obj,
        species=["H", "Cl"],
        rbf_gamma=0.1,
        function_title="test",
        round_result=6,
        n_jobs=5,
    )

    assert len(fitness_list) == 3
    assert isinstance(fitness_list[0], SpeciesSpecificSoapRbfSimFitness)
    assert fitness_list[0].db_title == "test_H_H"
    assert fitness_list[1].db_title == "test_H_Cl"
    assert fitness_list[2].db_title == "test_Cl_Cl"
    assert fitness_list[0].n_jobs == 5
    assert fitness_list[0].round_result == 6
    assert np.allclose(fitness_list[0].target_soap_features, np.array([1, 2, 3]))


def test_convert_ase_atoms_to_individual():
    ase_atoms = ase.Atoms()
    ind = Individual.from_ase(ase_atoms)
    # __eq__ method of ase atoms will still work
    assert ase_atoms == ind

    ase_atoms = ase.Atoms(
        ["H", "He"], [[0, 0, 0], [1, 1, 1]], cell=[1, 1, 1], pbc=[True, False, True]
    )
    ind = Individual.from_ase(ase_atoms)
    assert ase_atoms == ind
