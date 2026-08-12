=============
Fucrimodo CLI
=============

.. contents::
   :depth: 2
   :local:

Introduction
============

The command line interface is the main way to interact with the program 
FUCrIMODo. 
To use it simply run in a commandline::

    python fucrimodo

This will print the help message which will explain further details.
To get detailed help for a desired command, type::

    python fucrimodo <desired_command> help

More informations about these commands can also be found in the following
sections.

.. _cli-run:

Run Inversion with CLI
======================

One of the main use chases of the CLI is to perform an inversion on a 
descriptor specified in an input file 
(MISSING: LINK TO INFO ABOUT INPUT FILE)::

    python fucrimodo run PATH_TO_INPUT_FILE

.. _cli-analysis:

Analysis with CLI
=================

To analyse the results of a run that where saved to a specific results 
directory (MISSING: INFO ABOUT RESULTS DIR) a notebook can be created::

    python fucrimodo analyse notebook PATH_TO_RESULTS_DIR
    
The generation of plots, figures, tables or other results data can also be
done without the creation of notebooks. Just use::

    python fucrimodo analyse run PATH_TO_RESULTS_DIR

Detailed Information on Methods
===============================
.. toctree::
   :maxdepth: 1

   analyse
   run
