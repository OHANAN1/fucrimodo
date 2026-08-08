.. currentmodule:: fucrimodo.core

==========
Individual
==========

The individual is the solution of the optimization algorithm that should be
optimized. In the chase of `fucrimodo` the individual is an atomic structure.

.. autoclass:: fucrimodo.core.Individual
   :members:
   :show-inheritance:

===============
Fitness Storage
===============

.. currentmodule:: fucrimodo.core.individual

Each individual has a :class:`FitnessStorage` assigned as attribute :attr:`fitness` to store the fitness.

.. autoclass:: fucrimodo.core.individual.FitnessStorage
   :members:
   :show-inheritance:


==========
Population
==========

A list of individuals that get simultaneously optimized forms the population.

.. autoclass:: fucrimodo.core.Population
   :members:
   :undoc-members:
   :show-inheritance:
