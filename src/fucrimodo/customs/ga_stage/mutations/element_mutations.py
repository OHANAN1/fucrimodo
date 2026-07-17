from .abstract import Mutation
from fucrimodo.core.utils.closest_distances_class import CustomClosestDistances
from fucrimodo.core.modules import Individual
import ase.ga.standardmutations as ase_standard_mut
import ase
import ase.data as ase_data
import random
import numpy as np


class ReplaceAtomsMutation(Mutation):
    """
    Replace a random atom with an random atom species in the soap object.
    Atom has to be in the soap object species.

    Use soap_object.species to get the possible elements.
    """

    # NOTE: Maybe use the ase RandomElementMutation
    def __init__(
        self,
        possible_elements: list["str"],
        closest_distances: CustomClosestDistances,
        n_atoms_to_replace: int = 1,
        max_steps: int = 1000,
    ):
        self.possible_atomic_numbers: list[int] = [
            ase_data.atomic_numbers[atom] for atom in possible_elements
        ]
        self.n_atoms_to_replace = n_atoms_to_replace
        self.max_steps = max_steps
        self.closest_distances = closest_distances

    def perform_mutation(
        self,
        individual: Individual,
    ) -> Individual | None:

        if len(individual) < self.n_atoms_to_replace:
            return None

        atomic_numbers = individual.get_atomic_numbers()

        for _ in range(self.n_atoms_to_replace):
            random_index = random.choice(range(len(atomic_numbers)))
            selected_atomic_number = atomic_numbers[random_index]
            remaining_atomic_numbers = [
                number
                for number in self.possible_atomic_numbers
                if number != selected_atomic_number
            ]
            if len(remaining_atomic_numbers) == 0:
                return None
            else:
                atomic_numbers[random_index] = random.choice(remaining_atomic_numbers)

        individual.set_atomic_numbers(atomic_numbers)
        return individual


class AddAtomsMutation(Mutation):
    """
    Add a random atom to the individual.
    """

    def __init__(
        self,
        possible_elements: list[str],
        closest_distances: CustomClosestDistances,
        max_steps: int = 400,
        n_atoms_to_add: int = 1,
        only_same_species: bool = False,
    ):
        self.possible_atomic_numbers: list[int] = [
            ase_data.atomic_numbers[atom] for atom in possible_elements
        ]

        self.n_atoms_to_add = n_atoms_to_add
        self.closest_distances = closest_distances
        self.max_steps = max_steps
        self.only_same_species = only_same_species

    def finde_distant_point(
        self,
        positions: np.ndarray,
        target_distance: float,
        boundaries: tuple[tuple[float, float]],
        max_steps=100,
    ) -> np.ndarray | None:
        for _ in range(max_steps):
            candidate = np.array(
                [np.random.uniform(low, high) for low, high in boundaries]
            )

            distances = np.linalg.norm(positions - candidate, axis=1)

            if np.all(distances >= target_distance):
                self.logger.debug("Could not find a distant point")
                return candidate

        return None

    def perform_mutation(
        self,
        individual: Individual,
    ) -> Individual | None:

        positions = individual.get_positions()

        for _ in range(self.n_atoms_to_add):

            random_atomic_number = random.choice(self.possible_atomic_numbers)
            if self.only_same_species:
                unique_numbers = set(individual.get_atomic_numbers())
                if len(unique_numbers) == 1:
                    random_atomic_number = individual.get_atomic_numbers()[0]
                else:
                    random_atomic_number = random.choice(list(unique_numbers))

            cell_len_ang = individual.cell.cellpar()
            boundaries = (
                (0, cell_len_ang[0]),
                (0, cell_len_ang[1]),
                (0, cell_len_ang[2]),
            )

            new_position = self.finde_distant_point(
                positions, 1.0, max_steps=100, boundaries=boundaries  # type: ignore
            )

            if new_position is None:
                return None

            else:
                individual.append(
                    ase.Atom(
                        random_atomic_number, position=new_position  # type: ignore
                    )
                )

        return individual


class PermutationMutation(Mutation):
    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        n_top: int | str = "all",
        prob: float = 0.5,
    ) -> None:
        self.n_top = n_top
        self.prob = prob
        self.closest_distances = closest_distances
        self.max_steps = 1

        if not isinstance(n_top, int) and not n_top == "all":
            raise ValueError("n_top has to be an integer or the string 'all'")

    def perform_mutation(self, individual: Individual) -> Individual | None:
        if len(individual.numbers) == 1:
            return None
        if len(np.unique(individual.numbers)) == 1:
            return None

        if self.n_top == "all":
            n_top = len(individual)
        else:
            n_top = self.n_top

        ase_permutation = ase_standard_mut.PermutationMutation(
            n_top=n_top,
            probability=self.prob,
            blmin=self.closest_distances,
            verbose=True,
        )

        offspring = individual
        mutant = ase_permutation.mutate(offspring)
        return mutant


class DeleteRandomAtomsMutation(Mutation):
    """
    Deletes a random atom from the individual.
    """

    def __init__(
        self,
        closest_distances: CustomClosestDistances,
        n_atoms_to_delete: int = 1,
    ):
        self.max_steps = 1
        self.n_atoms_to_delete = n_atoms_to_delete
        self.closest_distances = closest_distances

    def perform_mutation(self, individual: Individual) -> Individual | None:
        offspring = individual
        number_of_atoms = len(offspring.get_atomic_numbers())

        if number_of_atoms <= 1:
            return None

        for _ in range(self.n_atoms_to_delete):
            random_index = random.choice(range(number_of_atoms))
            offspring.pop(random_index)
            if len(offspring) == 1:
                return None

        return offspring
