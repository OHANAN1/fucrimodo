================
Break Conditions
================

The abstract class for the break condition is defined in
:mod:`fucrimodo.core.abstracts`. All implementations inherit from it. The
following break condition types are available

.. contents:: Available break condition types
    :local:
    :depth: 2

------------------------
Fitness Break Conditions
------------------------

Fitness break conditions stop the algorithm based on the fitness values of the
population.

.. autoclass:: fucrimodo.customs.break_conditions.MaxFitnessBreak
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: fucrimodo.customs.break_conditions.MinFitnessBreak
   :members:
   :undoc-members:
   :show-inheritance:

--------------------------
Generation Break Condition
--------------------------

.. autoclass:: fucrimodo.customs.break_conditions.GenerationBreak
   :members:
   :undoc-members:
   :show-inheritance:

----------------------
Logic Break Conditions
----------------------

Logic break conditions can be used together with other break conditions to get a
more complex behavior by using logic operations.


.. autoclass:: fucrimodo.customs.break_conditions.MultipleAndBreak
   :members:
   :undoc-members:
   :show-inheritance:


.. autoclass:: fucrimodo.customs.break_conditions.MultipleOrBreak
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: fucrimodo.customs.break_conditions.NotBreak
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: fucrimodo.customs.break_conditions.NeverBreak
   :members:
   :undoc-members:
   :show-inheritance:
