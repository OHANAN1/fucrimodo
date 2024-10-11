from .individual import Individual

class Population:
    def __init__(self, individuals: list[Individual]):
        """Stores the list of individuals in the population that is used
        in the optimization algorithm.
        """
        self.individuals = individuals
        self._generation = 0

    @property
    def individuals(self) -> list[Individual]:
        """The list of all individuals in the population.

        If new individuals are set, the generation number is incremented.
        """
        return self._individuals

    @individuals.setter
    def individuals(self, value: list[Individual]):
        self._individuals = value

        # Increment the generation number
        if not hasattr(self, "_generation"):
            self._generation = 0
        self._generation += 1

    @property
    def generation(self) -> int:
        """The generation number of the population."""
        return self._generation

    @property
    def size(self) -> int:
        """Returns the number of individuals in the population."""
        return len(self._individuals)
