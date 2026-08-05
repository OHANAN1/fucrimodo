===========================
The Multi-Stage Search Core
===========================

.. currentmodule:: fucrimodo.core

The :mod:`fucrimodo.core` module contains the framework and utilities to implement the multi-stage
search algorithm. At its center is the :class:`MultiStageSearch` class which
handles the data and time-keeping, as well as organizing and running stages.

.. toctree::
   :maxdepth: 1

   multi_stage_search

An multi stage ga workflow to invert the global SOAP descriptor is published in the paper [TODO: Add paper ref]. Its implementation can be found here: [TODO: Add ref to workflow implementation].

------------------
Abstract Framework
------------------

In addition to the main framework :mod:`fucrimodo.core.modules` adds the abstract :class:`Stage` to set up the individual parts of the optimization algorithm and some useful abstract base classes, needed in most optimization algorithms.
The classes in the module exist so a user can implement them for their specific use-chase. The module :mod:`fucrimodo.customs` includes multiple such examples.

.. currentmodule:: fucrimodo.core.modules

.. autosummary::
   :signatures: none

   Stage
   Individual
   Population
   BreakCondition
   FitnessFunction
   PopulationGenerator
   PopulationSelection

.. toctree::
   :maxdepth: 1

   modules

---------
Utilities
---------

.. currentmodule:: fucrimodo.core.utils

Apart from the abstract framework the core includes utilities for
performing the algorithm and interfacing with the framework.


.. toctree::
   :maxdepth: 2

   utils
