============
Core Modules
============

..
   .. automodule:: fucrimodo.core.modules
      :members:
      :undoc-members:
      :show-inheritance:

-----
Stage
-----

A stage should implement a full optimization algorithm. An example implementation for the Genetic algorithm can be found in :mod:`fucrimodo.customs.ga_stage`.


----------
Individual
----------

The individual is the solution of the optimization algorithm that should be
optimized.

.. autoclass:: fucrimodo.core.modules.Individual
   :members:
   :undoc-members:
   :show-inheritance:


A list of individuals then forms the population.

.. autoclass:: fucrimodo.core.modules.Population
   :members:
   :undoc-members:
   :show-inheritance:

