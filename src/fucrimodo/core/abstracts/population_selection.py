from abc import ABC, abstractmethod

from ..individual import Individual


class PopulationSelection(ABC):
    """Select a subset of individuals based on their properties.

    Population selection is used to select individuals from the population based
    on a selection strategy. E.g. in the genetic algorithms this is used to
    select individuals that get modified or used for the next generation.
    """

    @abstractmethod
    def select(self, individuals: list[Individual], n: int) -> list[Individual]:
        """Method that selects individuals from a given list of individuals.

        The selection strategy must be implemented in this method.  Note:
        Contrary to the name the method does not use the population but a list
        of individuals. This is often the desired chase, since the selected
        individuals often do not directly build the new population. To use
        :class:`Individuals` of a :class:`Population` do the following:

        .. code-block:: python

            selected_individuals = population_selection.select(population.individuals)

        :param individuals: A list of individuals that are used for
            the selection.
        :param n: The number of individuals that should be selected.

        :return: A list of individuals that were selected based on the
            implemented selection strategy.
        """
        pass

    def __repr__(self):
        class_name = self.__class__.__name__
        variables = vars(self)
        variables_str = ", ".join(f"{key}={value}" for key, value in variables.items())
        return f"{class_name}({variables_str})"
