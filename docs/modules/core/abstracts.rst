.. currentmodule:: fucrimodo.core.abstracts

.. autosummary::
   :signatures: none

   Stage
   BreakCondition
   FitnessFunction
   PopulationGenerator
   PopulationSelection

=====
Stage
=====

A stage should implement a full optimization algorithm. An example implementation for the Genetic algorithm can be found in :mod:`fucrimodo.customs.ga_stage`.

.. autoclass:: fucrimodo.core.abstracts.Stage
   :members:
   :show-inheritance:


================
Fitness Function
================

The fitness function is used to evaluate how well an individual fulfills a given objective of the optimization algorithm. This abstract class need to be implemented by the user or can use the examples found in :mod:`fucrimodo.customs.fitness_functions`.

Also look at :mod:`fucrimodo.core.utils.fitness_utils` for utilities that assign fitness values directly to an :attr:`Individual`.

.. autoclass:: fucrimodo.core.abstracts.FitnessFunction
   :members:
   :undoc-members:
   :show-inheritance:


====================
Population Generator
====================

To get a initial starting population or to seed an existing population with new individuals the population generator can be used. The user has to implement this abstract class or can use the examples found in :mod:`fucrimodo.customs.population_generators`.

.. autoclass:: fucrimodo.core.abstracts.PopulationGenerator
   :members:
   :undoc-members:
   :show-inheritance:


====================
Population Selection
====================

To select a population for example in a Genetic Algorithm the population selection is used. The user has to implement this abstract class or can use the examples found in :mod:`fucrimodo.customs.population_selections`.

.. autoclass:: fucrimodo.core.abstracts.PopulationSelection
   :members:
   :undoc-members:
   :show-inheritance:
