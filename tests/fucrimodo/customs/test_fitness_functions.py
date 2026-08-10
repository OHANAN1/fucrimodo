import numpy as np
import pytest
from fucrimodo.customs import fitness_functions as ff
from fucrimodo.core.abstracts import FitnessFunction
from fucrimodo.core.utils import CustomClosestDistances
from fucrimodo.core import Individual
from fucrimodo.customs.global_soap_target import GlobalSOAP


class TestPhysicalityFitness:
    @pytest.fixture
    def physicality_fitness(self):
        return ff.PhysicalityFitness(
            closest_distances=CustomClosestDistances(["H", "Na", "Cl", "O"], 0.8)
        )

    def test_evaluate_individuals(
        self, physicality_fitness: FitnessFunction, ind_slab, ind_molecule, ind_crystal
    ):
        f_slab = physicality_fitness.evaluate_individual(ind_slab)
        f_molecule = physicality_fitness.evaluate_individual(ind_molecule)
        f_crystal = physicality_fitness.evaluate_individual(ind_crystal)

        # All structures should have atoms far enough appart
        assert f_slab == 1.0
        assert f_molecule == 1.0
        assert f_crystal == 1.0

        # Create ind with close atoms
        bad_ind = Individual(["H", "H"], positions=[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
        f_bad = physicality_fitness.evaluate_individual(bad_ind)
        assert f_bad < 1.0

        worse_ind = Individual(["H", "H"], positions=[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
        f_worse = physicality_fitness.evaluate_individual(worse_ind)
        assert f_worse < f_bad

        worst_ind = Individual(["H", "H"], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        f_worst = physicality_fitness.evaluate_individual(worst_ind)
        assert f_worst < f_worse


def test__assign_features_to_individuals(periodic_soap_obj, ind_crystal, ind_slab):
    ind_crystal_copy = ind_crystal.copy()

    # Test on single individual
    assert ind_crystal.features is None
    ff._assign_features_to_individuals(periodic_soap_obj, [ind_crystal], n_jobs=1)
    assert type(ind_crystal.features) is np.ndarray

    # Test on multiple individual
    ff._assign_features_to_individuals(
        periodic_soap_obj, [ind_crystal_copy, ind_slab], n_jobs=1
    )
    assert type(ind_crystal_copy.features) is np.ndarray
    assert type(ind_slab.features) is np.ndarray


class TestSOAPSimilarityFitness:
    @pytest.fixture
    def soap_similarity_fitness(self, ind_crystal, periodic_soap_obj):
        target_features = periodic_soap_obj.create(ind_crystal)
        return ff.SoapRbfSimilarityFitness(
            target_soap_features=target_features,
            soap_object=periodic_soap_obj,
            rbf_gamma=1,
            db_title=None,
            n_jobs=1,  # Use only one job, since spawning processes is expencive
        )

    def test_evaluate_individual(
        self,
        soap_similarity_fitness: ff.SoapRbfSimilarityFitness,
        ind_crystal,
        ind_slab,
    ):
        # Features of target structures should be equal to target features
        # so fitness should be maximal
        f_crystal = soap_similarity_fitness.evaluate_individual(ind_crystal)
        assert type(ind_crystal.features) is np.ndarray, "Features should be set"
        assert f_crystal == pytest.approx(1.0)

        # Manipulate the structure very slightly
        pos = ind_crystal.positions
        pos[0] += 0.3
        ind_crystal.positions = pos
        ind_crystal.reset()

        f_crystal = soap_similarity_fitness.evaluate_individual(ind_crystal)
        assert f_crystal < 1.0

        # Slab has no overlapping atomic types therefore it should have
        # fitness of 0.
        f_slab = soap_similarity_fitness.evaluate_individual(ind_slab)
        assert f_slab == 0.0

    def test_evaluate_individuals(
        self,
        soap_similarity_fitness: ff.SoapRbfSimilarityFitness,
        ind_crystal,
        ind_slab,
    ):

        f_crystal, f_slab = soap_similarity_fitness.evaluate_individuals(
            [ind_crystal, ind_slab]
        )

        assert f_slab < f_crystal

    def test_repr_(self, soap_similarity_fitness: ff.SoapRbfSimilarityFitness):
        assert (
            soap_similarity_fitness.__repr__()
            == "SoapRbfSimilarityFitness(rbf_gamma=1)"
        )

    def test_automatic_db_title(
        self, soap_similarity_fitness: ff.SoapRbfSimilarityFitness
    ):
        assert soap_similarity_fitness.db_title == "SoapRbfSimilarityFitness"


class TestSpeciesSpecificSoapRbfSimFitness:
    @pytest.fixture()
    def species_specific_soap_rbf_sim_fitness(self, ind_crystal, periodic_soap_obj):
        target_features = periodic_soap_obj.create(ind_crystal)
        return ff.SpeciesSpecificSoapRbfSimFitness(
            target_soap_features=target_features,
            soap_object=periodic_soap_obj,
            rbf_gamma=1,
            db_title=None,
            species=("Na", "Cl"),
            n_jobs=1,  # Use only one job, since spawning processes is expencive
        )

    def test__get_rbf_sim_for_species(
        self,
        species_specific_soap_rbf_sim_fitness: ff.SpeciesSpecificSoapRbfSimFitness,
        ind_crystal,
        periodic_soap_obj: GlobalSOAP,
        ind_slab,
    ):
        # Test on single system
        rbf_sim = species_specific_soap_rbf_sim_fitness._get_rbf_sim_for_species(
            feature_vector_list=periodic_soap_obj.create([ind_slab])
        )
        # NOTE: Does not have to be 0., even tho candidate structure has no
        # overlapping species, bc some parts of the target soap slice are also 0
        assert rbf_sim < 1.0

        rbf_sim = species_specific_soap_rbf_sim_fitness._get_rbf_sim_for_species(
            feature_vector_list=periodic_soap_obj.create([ind_crystal])
        )
        assert rbf_sim == pytest.approx(1.0)

        # Test on multiple systems
        rbf_sim_list = species_specific_soap_rbf_sim_fitness._get_rbf_sim_for_species(
            feature_vector_list=periodic_soap_obj.create([ind_slab, ind_crystal])
        )
        assert rbf_sim_list[0] < 1.0
        assert rbf_sim_list[1] == pytest.approx(1.0)

    def test_evaluate_individual(
        self,
        species_specific_soap_rbf_sim_fitness: ff.SpeciesSpecificSoapRbfSimFitness,
        ind_crystal,
        ind_slab,
    ):
        f_crystal = species_specific_soap_rbf_sim_fitness.evaluate_individual(
            individual=ind_crystal
        )
        assert f_crystal == 1.0

        f_slab = species_specific_soap_rbf_sim_fitness.evaluate_individual(
            individual=ind_slab
        )
        assert f_slab < f_crystal

    def test_evaluate_individuals(
        self,
        species_specific_soap_rbf_sim_fitness: ff.SpeciesSpecificSoapRbfSimFitness,
        ind_crystal,
        ind_slab,
    ):
        # single individual
        f_crystal = species_specific_soap_rbf_sim_fitness.evaluate_individuals(
            individuals=[ind_crystal],
        )
        assert f_crystal == [1.0]

        # multiple individuals
        f_crystal, f_slab = species_specific_soap_rbf_sim_fitness.evaluate_individuals(
            individuals=[ind_crystal, ind_slab]
        )
        assert f_slab < f_crystal
