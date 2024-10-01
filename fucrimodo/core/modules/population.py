from .individual import Individual

class Population:
    def __init__(self, individuals: list[Individual]):
        """Stores the list of individuals in the population that is used
        in the optimization algorithm.
        """
        self.individuals = individuals

    @property
    def individuals(self) -> list[Individual]:
        """The list of all individuals in the population."""
        return self._individuals

    @individuals.setter
    def individuals(self, value: list[Individual]):
        self._individuals = value

    @property
    def size(self) -> int:
        """Returns the number of individuals in the population."""
        return len(self._individuals)
