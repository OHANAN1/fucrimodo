from .abstract import Mutation
from fucrimodo.core.modules import Individual, FitnessFunction
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.utils.custom_soap import CustomSOAP
import ase.ga.standardmutations as ase_standard_mut
import numpy as np


class RattleMutation(Mutation):
    """
    Moves n_max atoms randomly in all 3 directions.
    The maximal movement is defined by max_movement.
    This is then limited by the closest distance between atoms.
    The closest distance is calculated by the
    ase.ga.utilities.closest_distances_generator.

    The bool shuffle_when_n_top determines if the atoms are shuffled
    if n_top is not "all". If False, always the first n_top atoms are moved.

    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        n_top: int | str = "all",
        rattle_strength: float = 0.8,
        rattle_prop: float = 0.5,
        shuffle_when_n_top: bool = True
    ) -> None:
        self.n_top = n_top
        self.rattle_strength = rattle_strength
        self.rattle_prop = rattle_prop
        self.closest_distances = closest_distances
        self.max_steps = 1
        self.shuffle_when_n_top = shuffle_when_n_top

        if not isinstance(n_top, int) and not n_top == "all":
            raise ValueError(
                "n_top has to be an integer or the string 'all'"
            )

    def perform_mutation(self, crystal: Individual) -> Individual | None:
        if self.n_top == "all":
            n_top = len(crystal)
        else:
            n_top = self.n_top

            if self.shuffle_when_n_top:
                shuffled_crystal = crystal[
                    np.random.permutation(len(crystal))
                ]
                if isinstance(shuffled_crystal, Individual):
                    crystal = shuffled_crystal
                else:
                    raise ValueError("Shuffling did not work")

        ase_rattle = ase_standard_mut.RattleMutation(
            n_top=n_top,
            blmin=self.closest_distances,
            rattle_strength=self.rattle_strength,
            rattle_prop=self.rattle_prop,
            verbose=False
        )

        offspring = crystal
        mutant = ase_rattle.mutate(offspring)

        return mutant


class SmartRattleMutation(Mutation):
    """Tests `directions_to_test` random directions and moves the atom in the direction
    that results in the highest fitness.

    :param closest_distances: ClosestDistances object
    :param fitness_function: Fitness function that evaluates what is the best 
        crystal
    :param descriptor_object: Descriptor to calculate the feature vectors
        of all possible atom movements in parallel. This can safe computation
        time. If None, the fitness function takes care of the descriptor if 
        needed.
    :param directions_to_test: Number of random directions to test
    :param movement_step: Maximum movement step in Angstrom
    :param max_steps: Maximum number of steps to take
    """
    def __init__(
        self, 
        closest_distances: CustomClosestDistances,
        fitness_function: FitnessFunction,
        descriptor_object: CustomSOAP | None = None,
        directions_to_test: int = 3,
        movement_step: float = 0.1,
        max_steps: int = 10
        ):
        self.closest_distances = closest_distances
        self.max_steps = max_steps
        self.directions_to_test = directions_to_test
        self.movement_step = movement_step
        self.fitness_function = fitness_function
        self.descriptor_object = descriptor_object

    def perform_mutation(self, crystal: Individual) -> Individual | None:
        atom_index = np.random.randint(0, len(crystal) - 1)
        candidates = []
        for _ in range(self.directions_to_test):
            # Copy the crystal to not change the original
            candidate = crystal.copy()

            # rattle random atom in random direction
            candidate.positions[atom_index] += np.random.uniform(
                -self.movement_step, self.movement_step, 3
            )
            candidates.append(candidate)

        if self.descriptor_object is not None:
            feature_vectors = self.descriptor_object.create(candidates)
            for i, feature_vector in enumerate(feature_vectors):
                candidates[i].features = feature_vector

        # Evaluate fitness of all candidates
        fitnesses = self.fitness_function.evaluate_individuals(candidates)

        # Get the index of the best candidate and apply it to the crystal
        best_crystal_ind = np.argmax(fitnesses)

        new_positions = candidates[best_crystal_ind].positions
        crystal.positions[atom_index] = new_positions[atom_index]

        return crystal


class MirrorMutation(Mutation):
    """
    Mirrors the crystal along a random axis.
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        n_top: int | str = "all",
    ) -> None:
        self.closest_distances = closest_distances
        self.max_steps = 1
        self.n_top = n_top

        if not isinstance(n_top, int) and not n_top == "all":
            raise ValueError(
                "n_top has to be an integer or the string 'all'"
            )

    def perform_mutation(self, crystal: Individual) -> Individual | None:

        if self.n_top == "all":
            n_top = len(crystal)
        else:
            n_top = self.n_top

        ase_mirror = ase_standard_mut.MirrorMutation(
            blmin=self.closest_distances,
            n_top=n_top,
            verbose=False
        )

        offspring = crystal
        mutant = ase_mirror.mutate(offspring)

        return mutant
