==============
Analyse Module
==============

.. contents::
   :depth: 2
   :local:


TODO: Say that this was mainly build for GAStage, but other algorithms can implement data saving the same way as the GAStage. Then it can be analysed with the same classes. But to add new methods these need to be specifically changed in the corresponding module. Future plans are to make this more flexible but this is too complicated right now.
Say here exactly where users can find these things

Introduction
============

The analysis module can be used to analyse the data that was collected 
during a run. 

To make things consistent the files from the directory are first loaded into
the classes provided by :class:`RunResults`.
These classes can then be analysed by the classes in 
:mod:`AnalyseRun`.
Detailed information about these classes can be found here:

.. toctree::
   :maxdepth: 1

   analyse_run

Create Results Notebooks
========================

To ease the analysis and present the results, jupyter notebooks can be created
as described in the :ref:`cli-analysis`. 
An Example for such a results notebook can be found here:

.. .. toctree::
..    :maxdepth: 1
..
..    ../../example_results/example_run_results/results_notebook.ipynb

How these are created is completely adjustable.
[MISSING: information about how thez can be customized.]
