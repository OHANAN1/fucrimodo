import numpy as np
import pytest

from fucrimodo.customs.global_soap_target import GlobalSOAP


class TestGlobalSOAP:
    def test___init__(self):
        # Cannot initialize with averag 'off'
        with pytest.raises(AssertionError):
            GlobalSOAP(r_cut=2.0, n_max=2, l_max=2, species=["H"], average="off")  # type: ignore

        # Test that species get converted
        global_soap = GlobalSOAP(r_cut=2.0, n_max=2, l_max=2, species=[1, 2])
        assert global_soap.species == ["H", "He"]

    def test_get_init_params(self, periodic_soap_obj: GlobalSOAP):
        periodic_soap_copy = GlobalSOAP(**periodic_soap_obj.get_init_params())

        assert periodic_soap_copy.species == periodic_soap_obj.species
        assert periodic_soap_copy.r_cut == periodic_soap_obj.r_cut
        assert periodic_soap_copy.n_max == periodic_soap_obj.n_max
        assert periodic_soap_copy.l_max == periodic_soap_obj.l_max
        assert periodic_soap_copy.periodic == periodic_soap_obj.periodic
        assert periodic_soap_copy.average == periodic_soap_obj.average

    def test_create(self, periodic_soap_obj: GlobalSOAP, ind_slab, ind_crystal):
        # Create single system -> this creates np.ndarray
        single_features = periodic_soap_obj.create(ind_slab)
        assert type(single_features) == np.ndarray

        # Create system in list
        single_features_again = periodic_soap_obj.create([ind_slab])

        assert len(single_features_again) == 1
        assert np.allclose(single_features, single_features_again)

        # Create multiple systems
        multiple_features = periodic_soap_obj.create([ind_slab, ind_crystal])
        assert len(multiple_features) == 2
        assert multiple_features[0].shape == multiple_features[1].shape

    def test_get_present_species(self, periodic_soap_obj: GlobalSOAP, ind_crystal):

        species = periodic_soap_obj.get_present_species(
            periodic_soap_obj.create(ind_crystal)
        )
        assert len(species) == len(np.unique(ind_crystal.get_chemical_symbols()))

        for s in ind_crystal.get_chemical_symbols():
            assert s in species
