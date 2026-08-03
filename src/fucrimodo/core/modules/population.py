from .individual import Individual


class Population:
    """Object to store individuals.

    The individuals used during the run can be stored in the population, like a list.
    Additionally, the object has the :attr:`generation`. Whenever new individuals are
    assigned the generation value increases by one.
    """

    def __init__(self, individuals: list[Individual]):
        self.individuals = individuals
        self._generation = 0

    @property
    def individuals(self) -> list[Individual]:
        """The list of all individuals in the population.

        If new individuals are set, the generation number is incremented
        (see setter method).
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
        """The generation number of the population.

        The number is automatically increase by one if new
        individuals are assigned.
        """
        return self._generation

    @generation.setter
    def generation(self, value: int):
        self._generation = value

    @property
    def size(self) -> int:
        """Returns the number of individuals in the population."""
        return len(self._individuals)

    def __len__(self) -> int:
        return len(self._individuals)
