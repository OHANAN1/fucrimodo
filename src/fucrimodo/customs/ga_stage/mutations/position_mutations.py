from copy import deepcopy
from .abstract import Mutation
from fucrimodo.core.modules import Individual, FitnessFunction
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.customs.global_soap_target import GlobalSOAP, RBFSimilarity
import ase.ga.standardmutations as ase_standard_mut
import numpy as np


class RattleMutation(Mutation):
    """
    Moves len(atoms) - n_top atoms in a random directions.
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
        max_steps: int = 500,
        **kwargs,  # Only for backwards compatibility when params are changed
    ) -> None:
        self.n_top = n_top
        self.rattle_strength = rattle_strength
        self.rattle_prop = rattle_prop
        self.max_steps = max_steps
        self.closest_distances = closest_distances

        if not isinstance(n_top, int) and not n_top == "all":
            raise ValueError("n_top has to be an integer or the string 'all'")

    def perform_mutation(self, individual: Individual) -> Individual | None:
        # Set n_top to the desired value
        if self.n_top == "all":
            n_top = len(individual)
        else:
            n_top = int(self.n_top)

        # Make sure n_top is not larger than the number of atoms
        n_top = min(n_top, len(individual))

        positions = individual.get_positions().copy()
        indicees_to_rattle = np.random.choice(len(individual), n_top, replace=False)
        for i in indicees_to_rattle:
            positions[i] = positions[i] + np.random.normal(
                scale=self.rattle_strength, size=positions[i].shape
            )

        # Set the positions of the individual
        # This also respects the constraints set
        individual.set_positions(positions)

        # If constrains of Atoms made rattle movement not applicable
        # return None to signalize mutation failed
        if np.any(positions != individual.get_positions()):
            return None

        return individual


class SmartRattleMutation(Mutation):
    """Tests `directions_to_test` random directions and moves the atom in the direction
    that results in the highest fitness.

    :param closest_distances: ClosestDistances object
    :param fitness_function: Fitness function that evaluates what is the best
        individual
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
        descriptor_object: GlobalSOAP | None = None,
        directions_to_test: int = 3,
        max_movement: float = 0.1,
        max_steps: int = 10,
    ):
        self.closest_distances = closest_distances
        self.max_steps = max_steps
        self.directions_to_test = directions_to_test
        self.max_movement = max_movement
        self.fitness_function = fitness_function
        self.descriptor_object = descriptor_object

    def perform_mutation(self, individual: Individual) -> Individual | None:
        atom_index = np.random.randint(0, len(individual) - 1)
        candidates = []
        for _ in range(self.directions_to_test):
            # Copy the individual to not change the original
            candidate = individual.copy()

            # rattle positions in random direction
            positions = candidate.positions.copy()
            positions[atom_index] += np.random.uniform(
                -self.max_movement, self.max_movement, 3
            )

            # Set new positions (constraints apply)
            candidate.set_positions(positions)

            # If the candidate did not change due to constraints chance is
            # that the constraints apply for the selected atom_index
            # Return None since no change can be done with this atom_index
            if np.any(candidate.set_positions != positions):
                self.logger.debug(
                    "Could not mutate, since constrains are applied for "
                    f"selected atom_index {atom_index}."
                )

            candidates.append(candidate)

        if self.descriptor_object is not None:
            feature_vectors = self.descriptor_object.create(candidates)
            for i, feature_vector in enumerate(feature_vectors):
                candidates[i].features = feature_vector

        # Evaluate fitness of all candidates
        fitnesses = self.fitness_function.evaluate_individuals(candidates)

        # Get the index of the best candidate and apply it to the individual
        best_individual_ind = np.argmax(fitnesses)

        new_positions = candidates[best_individual_ind].positions
        individual.positions[atom_index] = new_positions[atom_index]

        return individual


class GradientRattleMutation(Mutation):
    """Picks a random atom and moves it in the direction of the gradient
    of the SOAPs rbf similarity.

    Currently only works with SOAP descriptor and the RBF similarity.

    :param closest_distances: ClosestDistances object
    :param rbf_similarity_obj: Object to calculate the rbf similarity
        to the target individual. Must have a descriptor_object set, so that
        the gradient can be calculated.
    :param max_steps: Maximum number of steps to take
    :param n_atoms_to_move: Number of atoms to move
    :param max_movement: Maximum movement step in Angstrom
    :param normalize_gradient: If True, the gradient is normalized before
        moving the atoms.
    :param n_jobs: Number of jobs to use for the calculation of the gradient
        of the soap descriptor. If set to -1, all reported CPUs are used.

    :raises AssertionError: If the descriptor object of the rbf_similarity_obj
        is not set.
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        rbf_similarity_obj: RBFSimilarity,
        max_steps: int = 10,
        n_atoms_to_move: int = 1,
        max_movement: float = 0.1,
        normalize_gradient: bool = True,
        n_jobs: int = 1,
    ):
        self.closest_distances = closest_distances
        self.max_steps = max_steps
        self.rbf_similarity_obj = rbf_similarity_obj
        self.n_atoms_to_move = n_atoms_to_move
        self.max_movement = max_movement
        self.normalize_gradient = normalize_gradient
        self.n_jobs = n_jobs

        # Check if the rbf_similarity_obj has a descriptor
        # If it is not set, the derivative cannot be calculated
        assert (
            rbf_similarity_obj.descriptor_object is not None
        ), "The descriptor object of the rbf_similarity_obj must be set."

    def perform_mutation(self, individual: Individual) -> Individual | None:
        # adjust n_atoms_to_move if it is larger than the number of atoms
        if self.n_atoms_to_move > len(individual):
            n_atoms_to_move = len(individual)
        else:
            n_atoms_to_move = self.n_atoms_to_move

        # Pick random atoms to move, all atoms indicees are only picked once
        atomic_indices = np.random.choice(
            np.arange(len(individual)), size=n_atoms_to_move, replace=False
        )

        # Calculate the gradient
        _, gradient = self.rbf_similarity_obj.derivative(
            individual, include=atomic_indices.tolist(), kwargs={"n_jobs": self.n_jobs}
        )

        positions = individual.positions.copy()

        # Move the atoms
        for i in range(len(atomic_indices)):
            # Normalize the gradient
            if self.normalize_gradient:
                norm = np.linalg.norm(gradient[i])
            else:
                norm = 1

            if norm != 0:
                gradient[i] = [x / norm for x in gradient[i]]
            else:
                gradient[i] = [0, 0, 0]

            # Move the atomic positions with a random factor in the direction
            # of the gradient
            positions[atomic_indices[i]] += [
                x * np.random.uniform(0, self.max_movement) for x in gradient[i]
            ]

        # Set the new positions (constraints apply)
        individual.set_positions(positions)

        # If no positions could be changed return None
        # To symbolize that mutation failed
        if np.all(positions != individual.positions):
            return None

        return individual


class MirrorMutation(Mutation):
    """
    Mirrors the individual along a random axis.
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
            raise ValueError("n_top has to be an integer or the string 'all'")

    def perform_mutation(self, individual: Individual) -> Individual | None:

        if self.n_top == "all":
            n_top = len(individual)
        else:
            n_top = self.n_top

        ase_mirror = ase_standard_mut.MirrorMutation(
            blmin=self.closest_distances, n_top=n_top, verbose=False
        )

        offspring = individual
        mutant = ase_mirror.mutate(offspring)

        return mutant
