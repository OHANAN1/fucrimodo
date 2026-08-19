import ase
import numpy as np
import pytest

from fucrimodo.customs.population_generators import (
    RandomSampleCrystalPopulation,
    create_random_crystal,
)


def test_create_random_structure(closest_distances):
    crystal = create_random_crystal(
        present_species=["H"],
        n_atoms=2,
        closest_distances=closest_distances,
        possible_space_groups=range(1, 231),
        n_tries_composition=1000,
        seed=1,
        sort_composition_descending=True,
    )

    assert type(crystal) == ase.Atoms
    assert len(crystal) == 2
    assert np.unique(crystal.get_chemical_symbols()).tolist() == ["H"]
    assert crystal.pbc is True or all(crystal.pbc == [True, True, True])

    # Check if running the same seed creates different structure
    same_crystal = create_random_crystal(
        present_species=["H"],
        n_atoms=2,
        closest_distances=closest_distances,
        possible_space_groups=range(1, 231),
        n_tries_composition=1000,
        seed=1,
        sort_composition_descending=True,
    )
    assert crystal == same_crystal

    # Check if running with different seed creates different structure
    different_crystal = create_random_crystal(
        present_species=["H"],
        n_atoms=2,
        closest_distances=closest_distances,
        possible_space_groups=range(1, 231),
        n_tries_composition=1000,
        seed=2,
        sort_composition_descending=True,
    )
    assert crystal != different_crystal


class TestRandomSampleCrystalPopulation:

    @pytest.fixture()
    def random_sample_crystal_population(
        self, periodic_soap_obj, ind_slab, closest_distances, example_fitness
    ):
        target_features = periodic_soap_obj.create(ind_slab, n_jobs=1)
        return RandomSampleCrystalPopulation(
            soap_obj=periodic_soap_obj,
            target_features=target_features,
            closest_distances=closest_distances,
            n_atoms=2,
            fitness_functions=example_fitness,
            n_jobs=1,
            exclude_space_groups=[1],
            rng=np.random.default_rng(42),
            n_samples=10,
        )

    def test___init__(
        self, random_sample_crystal_population: RandomSampleCrystalPopulation
    ):
        assert len(random_sample_crystal_population.possible_space_groups) > 0

        # Present species are only those who actually have non-zero features in
        # the target features
        assert hasattr(random_sample_crystal_population, "present_species")
        assert len(random_sample_crystal_population.present_species) == 1
        assert "H" in random_sample_crystal_population.present_species

    def test__get_possible_space_groups(
        self, random_sample_crystal_population: RandomSampleCrystalPopulation
    ):
        possible_space_groups = (
            random_sample_crystal_population._get_possible_space_groups(n_atoms=1)
        )

        # Excluded 1
        assert 1 not in possible_space_groups

    def test_generate_individuals(
        self, random_sample_crystal_population: RandomSampleCrystalPopulation
    ):
        # random_sample_crystal_population_copy = deepcopy(
        #     random_sample_crystal_population
        # )

        # Should generate same individuals if run again
        inds_1 = random_sample_crystal_population.generate_individuals(3)

        inds_2 = random_sample_crystal_population.generate_individuals(3)

        assert inds_1 == inds_2

        # Also generating parallel leads to same results
        random_sample_crystal_population.n_jobs = 4
        inds_3 = random_sample_crystal_population.generate_individuals(3)
        assert inds_2 == inds_3

        # Get different results if seeds change
        random_sample_crystal_population._sample_seeds += 100
        inds_4 = random_sample_crystal_population.generate_individuals(3)
        assert inds_3 != inds_4
