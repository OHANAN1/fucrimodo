=================
Fitness Functions
=================


The abstract class for the mutation is defined in
:mod:`fucrimodo.core.abstracts`. All implementations inherit from it. The
following fitness function types are available:

.. currentmodule:: fucrimodo.customs.fitness_functions

.. contents:: Available fitness types
    :local:
    :depth: 1

-------------------
Physicality fitness
-------------------

Fitness evaluating the physical feasibility of the individuals.

.. autoclass:: fucrimodo.customs.fitness_functions.PhysicalityFitness
   :members:
   :undoc-members:
   :show-inheritance:

------------------
Similarity fitness
------------------

Fitness functions evaluating the similarity of the SOAP features of the
individuals to the target.

.. autoclass:: fucrimodo.customs.fitness_functions.SoapRbfSimilarityFitness
   :members:
   :undoc-members:
   :show-inheritance:


.. autoclass:: fucrimodo.customs.fitness_functions.SpeciesSpecificSoapRbfSimFitness
   :members:
   :undoc-members:
   :show-inheritance:
