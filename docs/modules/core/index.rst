=======================
Fucrimodo Core Overview
=======================

.. currentmodule:: fucrimodo.core

The :mod:`fucrimodo.core` module contains the framework and utilities to implement the multi-stage search algorithm. At its center is the :class:`MultiStageSearch` class which handles the data and time-keeping, as well as organizing and running stages.

.. toctree::
   :maxdepth: 1

   multi_stage_search

A multi stage GA workflow to invert the global SOAP descriptor is published in the paper [TODO: Add paper ref]. Its implementation can be found here: [TODO: Add ref to workflow implementation].

---------
Solutions
---------

Possible candidate solutions of the optimization algorithms are so-called individuals. These get optimized to fullfill a given objective which is measured with the fitness value. All individuals used in fucrimodo are atomic structures. A collection of multiple individuals that simultaneously get optimized is called the population.

.. toctree::
   :maxdepth: 1

   solutions

--------------
Core Abstracts
--------------

The module :mod:`abstracts` defines how individual parts of the optimization algorithm should be implemented. All classes need to be implemented by the used for their specific use-chase. Multiple example implementations can be found in the module :mod:`fucrimodo.customs`.

.. toctree::
   :maxdepth: 1

   abstracts

--------------
Core Utilities
--------------

.. currentmodule:: fucrimodo.core.utils

Apart from the abstract framework the core includes utilities for performing the algorithm and interfacing with the framework.

.. toctree::
   :maxdepth: 2

   utils
