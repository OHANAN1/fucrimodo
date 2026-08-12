=====================
Population Selections
=====================

The abstract class for population selection is implemented in
:mod:`fucrimodo.core.abstracts`. All implementations inherit from it.  The
following selection types are available:


.. contents:: Available population selection types
    :local:
    :depth: 1


.. currentmodule:: fucrimodo.customs.population_selections

-----------------
NSGA-II Selection
-----------------

.. autoclass:: fucrimodo.customs.population_selections.NSGA2Selection
   :members:
   :undoc-members:
   :show-inheritance:

----------------
Random Selection
----------------

.. autoclass:: fucrimodo.customs.population_selections.RandomSelection
   :members:
   :undoc-members:
   :show-inheritance:

---------------------
Tournament Selections
---------------------

There are two types of tournament selections. The :class:`TournamentSelection`
considers only one fitness value, while the :class:`TournamentDCDSelection`
considers multiple fitness values at once.

.. autoclass:: fucrimodo.customs.population_selections.TournamentSelection
   :members:
   :undoc-members:
   :show-inheritance:


.. autoclass:: fucrimodo.customs.population_selections.TournamentDCDSelection
   :members:
   :undoc-members:
   :show-inheritance:
