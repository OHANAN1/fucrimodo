=================
Fucrimodo Customs
=================

This module implements concrete classes and functions that are used in an
multi-stage genetic algorithm. All components were used to create the workflow
discussed in the paper [TODO: Add link to paper]. Below more information about the
Genetic Algorithm and how it is implemented in a genetic algorithm stage can be found.

The :mod:`fucrimodo.customs` module includes the following submodules:

.. toctree::
   :maxdepth: 1

   fitness_functions
   population_generators
   population_selections
   break_conditions
   global_soap_target
   utils

-----------------
Genetic Algorithm
-----------------

In addition to the pre-mentioned submodules, :mod:`fucrimodo.customs.ga_stage`
provides a concrete implementation of a genetic algorithm.

.. currentmodule:: fucrimodo.customs.ga_stage.ga_stage

A genetic algorithm (GA) is a population-based, stochastic optimization method
inspired by darvinian evolution theory. A GA maintains a *population* of
candidate solutions (*individuals*) and improves them over successive
*generations* by applying operators loosely modeled on natural selection and
genetic inheritance:

1. **Evaluation** – Every individual is scored by one or more *fitness
   functions*. When several objectives are used, their values can be
   combined with individual weights to form a single fitness measure.
2. **Parent selection** – A subset of the population is chosen to become
   parents, typically favoring individuals with better fitness while still
   preserving some diversity.
3. **Crossover (recombination)** – Pairs of parents are combined to produce
   offspring that inherit traits from both, with a given *crossover
   probability*.
4. **Mutation** – Offspring are randomly perturbed with a given *mutation
   probability* to introduce new variation and avoid premature convergence.
5. **Survivor selection** – The next generation is chosen from the pool of
   parents and offspring, again based on fitness (and sometimes diversity).
6. **Termination** – The loop of selection, variation and evaluation repeats
   until a break condition is met, e.g. a maximum number of generations, a
   convergence criterion, or a fitness threshold.

Because crossover and mutation are stochastic and only accepted individuals
survive, the population as a whole tends to drift toward higher-fitness
regions of the search space, while still being able to escape local optima
through random variation.

This package's :class:`GAStage`/:class:`GeneticAlgorithm` implementation
follows this general scheme: a :class:`Population` of individuals is scored
via configurable fitness functions, evolved through weighted pools of
crossover and mutation operators, and reduced back to size via configurable
parent and survivor selection strategies, with a hall of fame keeping track
of the best individuals found. The algorithm is terminated based on a break
condition controlling termination.

Several design decisions are inspired by two well-established GA
frameworks:

- `DEAP <https://deap.readthedocs.io/>`_ (Distributed Evolutionary Algorithms
  in Python), a general-purpose evolutionary computation framework that
  provides flexible building blocks (individuals, fitness, statistics,
  logbooks, hall of fame, etc.) rather than fixed algorithm implementations,
  letting users compose their own evolutionary loop. This project reuses several of
  DEAP's ``tools`` components (e.g. ``HallOfFame``, ``Statistics``,
  ``Logbook``) directly.
- `ASE_GA <https://dtu-energy.github.io/ase-ga/>`_, a genetic algorithm
  framework built on top of the Atomic Simulation Environment (ASE) for
  global optimization of atomic structures, offering configurable
  populations, pairing (crossover) operators, mutations and stopping
  conditions. Multiple crossovers and mutations are directly taken from
  this project.


Details about the implementation of the :class:`GAStage` as well as the
modification operators :class:`Mutation` and :class:`Crossover` can be found
here:

.. toctree::
   :maxdepth: 1

   ga_stage
   mutations
   crossovers
