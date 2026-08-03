import ase
from operator import mul, truediv
from collections.abc import Sequence
import sys
import numpy as np
import datetime


class FitnessStorage(object):
    """Storage of fitness, inspired by the DEAP library.

    More information can be found in the DEAP documentation for the
    :class:'deap.base.Fitness' class. The main difference is that the
    FitnessStorage class can be initialized directly with a list of weights for
    the fitness values and does not depend on the DEAP creator module.
    Everything else works the same way as the DEAP Fitness class, to ensure
    compatibility with the DEAP framework.

    :param weights: A sequence of weights that are associated with the fitness
        values. The weights are used to calculate the weighted fitness values.
    """

    def __init__(self, weights=None):
        self.weights = weights
        self.wvalues = ()
        if self.weights is not None and not isinstance(self.weights, Sequence):
            raise TypeError(
                "Attribute weights of %r must be a sequence." % self.__class__
            )

    # TODO: Make this code more python like and consistent
    def getValues(self):
        if self.weights is None:
            return self.wvalues
        else:
            return tuple(map(truediv, self.wvalues, self.weights))

    def setValues(self, values):
        if self.weights is None:
            self.wvalues = values
        else:
            assert len(values) == len(
                self.weights
            ), f"Assigned values have not the same length than fitness weights. weight-lenght: {len(self.weights)}"
            try:
                self.wvalues = tuple(map(mul, values, self.weights))
            except TypeError:
                _, _, traceback = sys.exc_info()
                raise TypeError(
                    "Both weights and assigned values must be a "
                    "sequence of numbers when assigning to values of "
                    "%r. Currently assigning value(s) %r of %r to a "
                    "fitness with weights %s."
                    % (self.__class__, values, type(values), self.weights)
                ).with_traceback(traceback)

    def delValues(self):
        self.wvalues = ()

    values = property(
        getValues,
        setValues,
        delValues,
        (
            "Fitness values. Use directly ``individual.fitness.values = values`` "
            "in order to set the fitness and ``del individual.fitness.values`` "
            "in order to clear (invalidate) the fitness. The (unweighted) fitness "
            "can be directly accessed via ``individual.fitness.values``."
        ),
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

    def __str__(self):
        """Return the values of the Fitness object."""
        return str(self.values if self.valid else tuple())

    def __repr__(self):
        """Return the Python code to build a copy of the object."""
        return "%s.%s(%r)" % (
            self.__module__,
            self.__class__.__name__,
            self.values if self.valid else tuple(),
        )


class Individual(ase.Atoms):
    """
    An individual of the population. Inherits from :class:`ase.Atoms` and
    is initialized with the same arguments. However, it has additional
    attibutes: fitness, fitness_weights, features.

    fitness values and additional information.

    :param args: Arguments for the :class:`ase.Atoms` class.
        Common args are :data:'symbols', :data:'positions', :data:'cell',
        :data:'pbc', etc.
    :param kwargs: Keyword arguments for the :class:`ase.Atoms` class.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.info = {}
        self._creation_time = datetime.datetime.now()

    @property
    def fitness(self) -> FitnessStorage:
        """A storage for the fitness values of the individual.

        Uses the :class:'deap.base.Fitness' class with the weights set to the
        fitness_weights.
        The fitness values are stored in the 'values' attribute of the
        fitness object. Please set up the storage with the
        method `set_up_fitness_storage`. If not set up an empty fitnessStorage
        will be returned i.e. no values can be entered.
        Example:

        .. code-block:: python

            individual = Individual(ase.Atoms())
            individual.fitness.weights = (1., 0.5)
            individual.fitness.values = (1., 2.)
            print(individual.fitness.values)
            # (1.0, 2.0)

            individual.fitness.values = (2., 3.)
            print(individual.fitness.values)
            # (2.0, 3.0)
        """
        if not hasattr(self, "_fitness"):
            # Generate an empty fitness storage
            self._fitness = FitnessStorage(weights=None)

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
    def features(self) -> np.ndarray | None:
        if not hasattr(self, "_features"):
            return None
        return self._features

    @features.setter
    def features(self, value: np.ndarray | None):
        self._features = value

    @property
    def creation_time(self) -> datetime.datetime:
        return self._creation_time

    def reset(self):
        self._features = None
        self._creation_time = datetime.datetime.now()
        del self.fitness.values
