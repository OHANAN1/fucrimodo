import datetime
import time

import ase
import numpy as np
import pytest

from fucrimodo.core.individual import FitnessStorage, Individual


class TestFitnessStorage:
    def test_init_default(self):
        f = FitnessStorage()
        assert f.weights is None
        assert f.wvalues == ()

    def test_init_with_weights(self):
        f = FitnessStorage(weights=(1.0, -1.0))
        assert f.weights == (1.0, -1.0)
        assert f.wvalues == ()

    def test_set_and_get_values_weighted(self):
        f = FitnessStorage(weights=(2.0, -1.0))
        f.values = (1.0, 3.0)
        # wvalues are weighted: (1*2, 3*-1)
        assert f.wvalues == (2.0, -3.0)
        # getValues divides back out the weights
        assert f.values == (1.0, 3.0)

    def test_set_values_no_weights(self):
        f = FitnessStorage()
        f.values = (1.0, 2.0)
        # Without weights the raw values are stored directly
        assert f.wvalues == (1.0, 2.0)
        assert f.values == (1.0, 2.0)

    def test_set_values_wrong_length_raises(self):
        f = FitnessStorage(weights=(1.0, 1.0))
        with pytest.raises(AssertionError):
            f.values = (1.0,)

    def test_set_values_non_numeric_raises(self):
        f = FitnessStorage(weights=(1.0, 1.0))
        with pytest.raises(TypeError):
            f.values = ("a", "b")

    def test_del_values(self):
        f = FitnessStorage(weights=(1.0,))
        f.values = (5.0,)
        del f.values
        assert f.wvalues == ()
        assert not f.valid

    def test_del_weights(self):
        f = FitnessStorage(weights=(1.0,))
        f.values = (5.0,)
        del f.weights
        assert f.values == ()
        assert f.wvalues == ()
        assert not f.valid

    def test_overwriting_weights_deletes_values(self):
        f = FitnessStorage(weights=(1.0,))
        f.values = (5.0,)
        f.weights = (1.0, 2.0)
        assert f.values == ()
        assert f.wvalues == ()
        assert not f.valid

    def test_valid_false_when_empty(self):
        assert FitnessStorage().valid is False

    def test_valid_true_when_set(self):
        f = FitnessStorage(weights=(1.0,))
        f.values = (1.0,)
        assert f.valid is True

    def test_dominates_true(self):
        a, b = FitnessStorage((1.0, 1.0)), FitnessStorage((1.0, 1.0))
        a.values = (2.0, 2.0)
        b.values = (1.0, 2.0)
        assert a.dominates(b) is True

    def test_dominates_false_when_equal(self):
        a, b = FitnessStorage((1.0, 1.0)), FitnessStorage((1.0, 1.0))
        a.values = (1.0, 1.0)
        b.values = (1.0, 1.0)
        assert a.dominates(b) is False

    def test_dominates_false_when_worse(self):
        a, b = FitnessStorage((1.0, 1.0)), FitnessStorage((1.0, 1.0))
        a.values = (1.0, 0.0)
        b.values = (1.0, 2.0)
        assert a.dominates(b) is False

    def test_dominates_with_obj_slice(self):
        a, b = FitnessStorage((1.0, 1.0)), FitnessStorage((1.0, 1.0))
        a.values = (2.0, 0.0)
        b.values = (1.0, 5.0)
        # Only look at the first objective
        assert a.dominates(b, obj=slice(0, 1)) is True

    def test_equality_and_inequality(self):
        a, b = FitnessStorage(), FitnessStorage()
        a.values = (1.0, 2.0)
        b.values = (1.0, 2.0)
        assert a == b
        assert not (a != b)
        b.values = (1.0, 3.0)
        assert a != b

    def test_ordering(self):
        a, b = FitnessStorage(), FitnessStorage()
        a.values = (1.0,)
        b.values = (2.0,)
        assert a < b
        assert a <= b
        assert b > a
        assert b >= a

    def test_hash_consistency(self):
        a, b = FitnessStorage(), FitnessStorage()
        a.values = (1.0, 2.0)
        b.values = (1.0, 2.0)
        assert hash(a) == hash(b)

    def test_str_valid_and_invalid(self):
        f = FitnessStorage(weights=(1.0,))
        assert str(f) == str(())  # invalid -> empty tuple
        f.values = (3.0,)
        assert str(f) == str((3.0,))

    def test_repr_roundtrip(self):
        f = FitnessStorage(weights=(1.0,))
        f.values = (3.0,)
        assert "FitnessStorage" in repr(f)


class TestIndividual:
    @pytest.fixture
    def atoms(self):
        return Individual("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])

    def test_is_ase_atoms(self, atoms):
        assert isinstance(atoms, ase.Atoms)
        assert len(atoms) == 3

    def test_default_attributes(self, atoms):
        assert atoms.info == {}
        assert isinstance(atoms.creation_time, datetime.datetime)
        assert atoms.features is None

    def test_lazy_fitness_initialization(self, atoms):
        # Calling fitness without setting returns an empty
        # fitness object
        assert atoms.fitness.weights == None
        assert atoms.fitness.values == ()
        assert not atoms.fitness.valid

    def test_set_fitness_values(self, atoms: Individual):
        atoms.fitness.weights = (1.0,)
        atoms.fitness.values = (1.0,)
        assert atoms.fitness.values == (1.0,)
        assert atoms.fitness.valid

        atoms.fitness.values = (2.0,)
        assert atoms.fitness.values == (2.0,)
        assert atoms.fitness.valid

    def test_set_new_fitness(self, atoms: Individual):
        atoms.fitness.weights = (1.0,)
        assert not atoms.fitness.valid
        atoms.fitness.values = (1.0,)
        assert atoms.fitness.valid

        # Test that fitness storage will be overwritten
        atoms.fitness.weights = (1.0, 1.0)
        atoms.fitness.values = (1.0, 2.0)
        assert atoms.fitness.values == (1.0, 2.0)
        assert atoms.fitness.valid

        # Cannot set new fitness if its not the same shape as fitness
        with pytest.raises(AssertionError):
            atoms.fitness.values = (1.0, 2.0, 3.0)

    def test_info_setter(self, atoms):
        atoms.info = {"mutation": "swap"}
        assert atoms.info["mutation"] == "swap"

    def test_features_setter(self, atoms):
        feats = np.array([1.0, 2.0, 3.0])
        atoms.features = feats
        np.testing.assert_array_equal(atoms.features, feats)

    def test_reset(self, atoms):
        atoms.fitness.weights = (1.0,)
        atoms.fitness.values = (5.0,)
        atoms.features = np.array([1.0])
        old_time = atoms.creation_time

        time.sleep(0.001)
        atoms.reset()

        assert atoms.features is None
        assert not atoms.fitness.valid
        assert atoms.creation_time > old_time

    def test_from_ase_atoms(self, atoms):
        # Empty atoms
        atoms = ase.Atoms()

        ind = Individual.from_ase(atoms)
        assert ind == atoms
