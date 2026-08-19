import datetime
import sys
from operator import mul, truediv
from typing import Self

import ase
import numpy as np


class FitnessStorage(object):
    """Storage of fitness and their weight values.

    Implementation is inspired by the `DEAP library <https://github.com/deap/deap>`__.
    FitnessStorages can be compared with each other via the comparison operators.
    This compares the weighted fitness values :attr:`wvalues` lexicographically.

    More information can be found in the DEAP documentation for the
    :class:`deap.base.Fitness` class. The main difference is that the
    FitnessStorage class can be initialized directly with a list of weights for
    the fitness values and does not depend on the DEAP creator module.
    Everything else works the same way as the DEAP Fitness class, to ensure
    compatibility with the DEAP framework.

    :param weights: A sequence of weights that are associated with the fitness
        values. The weights are used to calculate the weighted fitness values.
    """

    def __init__(self, weights: tuple | None = None):
        self._weights = weights
        self.wvalues = ()

    @property
    def weights(self):
        """Weights of each of the fitness values.

        Setting new weights deletes the current :attr:`values`.
        """
        return self._weights

    @weights.setter
    def weights(self, weights: tuple | None):
        # reset the values
        del self.values

        self._weights = weights

    @weights.deleter
    def weights(self):
        self._weights = None

        # also delete values
        del self.values

    @property
    def values(self):
        """Fitness values.

        Use directly ``individual.fitness.values = values`` in order to set the
        fitness and ``del individual.fitness.values`` in order to clear
        (invalidate) the fitness. The (unweighted) fitness can be directly
        accessed via ``individual.fitness.values``.
        """
        if self.weights is None:
            return self.wvalues
        else:
            return tuple(map(truediv, self.wvalues, self.weights))

    @values.setter
    def values(self, values):
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

    @values.deleter
    def values(self):
        self.wvalues = ()

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
        """Assess if a fitness is valid or not.

        Valid means that fitness values are assigned
        """
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
    """A solution to the optimization problem.

    The individual inherits from :class:`ase.Atoms` and is initialized with the
    same arguments. However, it has the additional attibutes :attr:`fitness`,
    :attr:`features` and :attr:`creation_time`.

    :param args: Arguments for the :class:`ase.Atoms` class.
        Common args are :data:`symbols`, :data:`positions`, :data:`cell`,
        :data:`pbc`, etc.
    :param kwargs: Keyword arguments for the :class:`ase.Atoms` class.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.info = {}
        self._creation_time = datetime.datetime.now()

    @property
    def fitness(self) -> FitnessStorage:
        """A storage for the fitness values of the individual.

        Stores fitness and fitness weights in a :class:`FitnessStorage`. The
        fitness values are stored in the 'values' attribute of the fitness
        object. Weights have to be set before the values are entered. The
        number of fitness values must always match the weights. Weights can be
        overwritten. The weighted fitnesses of different individuals can be
        compared via the comparison operators.
        Example:

        .. code-block:: python

            individual = Individual(ase.Atoms())
            individual.fitness.weights = (1.0, 0.5)
            individual.fitness.values = (1.0, 2.0)
            print(individual.fitness.values)
            # (1.0, 2.0)

            individual.fitness.values = (2.0, 3.0)
            print(individual.fitness.values)
            # (2.0, 3.0)

            individual.fitness.weights = (1.0)
            individual.fitness.values = (1.0)
            print(individual.fitness.values)
            # (1.0)

        """
        if not hasattr(self, "_fitness"):
            # Generate an empty fitness storage
            self._fitness = FitnessStorage(weights=None)

        return self._fitness

    @property
    def features(self) -> np.ndarray | None:
        """Descriptor features of the individual.

        Attribute can be used to store the features so they do not need to be
        recalculated for e.g. the similarity fitness.  Please assign features
        manually. Do not forget to reset the features if structural/chemical
        changes occure to the individual.
        """
        if not hasattr(self, "_features"):
            return None
        return self._features

    @features.setter
    def features(self, value: np.ndarray | None):
        self._features = value

    @property
    def creation_time(self) -> datetime.datetime:
        """Time the structure was first created or reset."""
        return self._creation_time

    def reset(self):
        """Reset the individual.

        Resets the attributes :attr:`features` and :attr:`fitness.values` and
        sets the :attr:`creation_time` to the current time.

        All other attributes, like e.g. :attr:`info` stay untouched.
        """
        self._features = None
        self._creation_time = datetime.datetime.now()
        del self.fitness.values

    @classmethod
    def from_ase(cls, atoms: ase.Atoms) -> Self:
        """Create an :class:`Individual` from an ASE :class:`ase.Atoms` object.

        :param atoms: The ASE :class:`ase.Atoms` object to convert.
        :return: A new :class:`Individual` populated with positions, cell,
            periodic boundary conditions, and chemical symbols.
        """
        return cls(
            positions=atoms.get_positions().copy(),
            cell=atoms.get_cell().copy(),
            pbc=atoms.pbc.copy(),
            symbols=atoms.get_chemical_symbols().copy(),
        )
