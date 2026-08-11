=====================
Population Generators
=====================

The abstract class for population generators is implemented in :mod:`fucrimodo.core.abstracts`. All implementations should inherit from it. Until now only one type of population generator is implemented.

--------------------------
Crystal Structure Sampling
--------------------------

.. autoclass:: fucrimodo.customs.population_generators.RandomSampleCrystalPopulation
   :members:
   :undoc-members:
   :show-inheritance:

To sample crystal structures the worker function is used:

.. automethod:: fucrimodo.customs.population_generators.create_random_crystal
