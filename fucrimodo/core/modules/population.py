from .individual import Individual

class Population:
    def __init__(self, individuals: list[Individual]):
        """
        A population of individuals.
        """
        self.individuals = individuals

    @property
    def individuals(self) -> list[Individual]:
        """
        A list of all individuals in the population.
        """
        return self._individuals

    @individuals.setter
    def individuals(self, value: list[Individual]):
        self._individuals = value

    @property
    def size(self) -> int:
        """
        The number of individuals in the population.
        """
        return len(self._individuals)
