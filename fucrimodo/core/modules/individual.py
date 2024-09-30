import ase
from deap import base, creator
from copy import deepcopy
from functools import partial
from operator import mul, truediv
from collections.abc import Sequence
import sys

class FitnessStorage(object):
    """Workaround for the DEAP Fitness class.

    More information can be found in the DEAP documentation for the 
    :class:'deap.base.Fitness' class.
    The main difference is that the FitnessStorage class can be initialized
    directly with a list of weights for the fitness values and does not depend
    on the DEAP creator module.
    Everything els works the same way as the DEAP Fitness class, to ensure 
    compatibility with the DEAP framework.

    :param weights: A sequence of weights that are associated with the fitness
        values. The weights are used to calculate the weighted fitness values.
    """
    def __init__(self, weights = None):
        self.weights = weights
        self.wvalues = ()
        if not isinstance(self.weights, Sequence):
            raise TypeError("Attribute weights of %r must be a sequence."
                            % self.__class__)

    def getValues(self):
        if self.weights is None:
            return self.wvalues
        else:
            return tuple(map(truediv, self.wvalues, self.weights))

    def setValues(self, values):
        if self.weights is None:
            self.wvalues = values
        else:
            assert len(values) == len(self.weights), "Assigned values have not the same length than fitness weights"
            try:
                self.wvalues = tuple(map(mul, values, self.weights))
            except TypeError:
                _, _, traceback = sys.exc_info()
                raise TypeError(
                    "Both weights and assigned values must be a "
                    "sequence of numbers when assigning to values of "
                    "%r. Currently assigning value(s) %r of %r to a "
                    "fitness with weights %s."
                    % (self.__class__, values, type(values),
                        self.weights)
                ).with_traceback(traceback)

    def delValues(self):
        self.wvalues = ()

    values = property(
        getValues, setValues, delValues,
        ("Fitness values. Use directly ``individual.fitness.values = values`` "
            "in order to set the fitness and ``del individual.fitness.values`` "
            "in order to clear (invalidate) the fitness. The (unweighted) fitness "
            "can be directly accessed via ``individual.fitness.values``.")
    )

    def dominates(self, other, obj=slice(None)):
        """Return true if each objective of *self* is not strictly worse than
        the corresponding objective of *other* and at least one objective is
        strictly better.

        :param obj: Slice indicating on which objectives the domination is
                    tested. The default value is `slice(None)`, representing
                    every objectives.
        """
        not_equal = False
        for self_wvalue, other_wvalue in zip(self.wvalues[obj], other.wvalues[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False
        return not_equal

    @property
    def valid(self):
        """Assess if a fitness is valid or not."""
        return len(self.wvalues) != 0

    def __hash__(self):
        return hash(self.wvalues)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __le__(self, other):
        return self.wvalues <= other.wvalues

    def __lt__(self, other):
        return self.wvalues < other.wvalues

    def __eq__(self, other):
        return self.wvalues == other.wvalues

    def __ne__(self, other):
        return not self.__eq__(other)

    # def __deepcopy__(self, memo):
    #     """Replace the basic deepcopy function with a faster one.
    #
    #     It assumes that the elements in the :attr:`values` tuple are
    #     immutable and the fitness does not contain any other object
    #     than :attr:`values` and :attr:`weights`.
    #     """
    #     copy_ = self.__class__()
    #     copy_.wvalues = self.wvalues
    #     return copy_

    def __str__(self):
        """Return the values of the Fitness object."""
        return str(self.values if self.valid else tuple())

    def __repr__(self):
        """Return the Python code to build a copy of the object."""
        return "%s.%s(%r)" % (self.__module__, self.__class__.__name__,
                              self.values if self.valid else tuple())


class Individual(ase.Atoms):
    """
    An individual in the population. Inherits from :class:`ase.Atoms`.
    Is initialized with the same arguments as the :class:`ase.Atoms` class.
    Different attributes can be applied to the individual:

        - fitness: A storage for the fitness values of the individual.
        - features: A dictionary with additional features of the individual.
            For example, the SOAP features of the individual.
    """
    def __init__(
        self, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.info = {}

    def reset(self):
        self.features = None
        del self.fitness.values

    @property
    def fitness_weights(self) -> Sequence[float] | None:
        return self._fitness_weights

    @fitness_weights.setter
    def fitness_weights(self, value: Sequence[float] | None):
        """Set the weights for the fitness values.

        Automatically resets the FitnessStorage object to delete old values
        and set the new weights.
        """
        self._fitness = FitnessStorage(value)
        self._fitness_weights = value

    @property
    def fitness(self) -> FitnessStorage:
        """A storage for the fitness values of the individual.

        Uses the :class:'deap.base.Fitness' class with the weights set to the
        fitness_weights.
        The fitness values are stored in the 'values' attribute of the
        fitness object.
        Example:

        .. code-block:: python
            
            individual = Individual(ase.Atoms())
            individual.fitness = (1.0, 2.0, 3.0)
            fitness.values = (1.0, 2.0, 3.0)
            print(fitness.values)
            # (1.0, 2.0, 3.0)

        """
        if not hasattr(self, "_fitness"):
            self._fitness = FitnessStorage(self.fitness_weights)

        return self._fitness

    @property
    def info(self) -> dict:
        """A dictionary that can be used to store additional information about
        the individual.

        For example, the mutation that was applied to the individual.
        Is NOT reset when the individual is reset.
        Must be reset manually or overwritten if needed.
        """
        return self._info

    @info.setter
    def info(self, value: dict):
        self._info = value

    @property
    def features(self) -> dict | None:
        return self._features

    @features.setter
    def features(self, value: dict | None):
        self._features = value
