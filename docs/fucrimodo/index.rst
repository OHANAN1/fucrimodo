=============
Documentation
=============

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    core/index
    customs/index
    analysis/index
    cli/index
    lab/index


.. currentmodule:: fucrimodo


FUCrIMODo consists of 5 main parts.

    - **Basic Framework** - :doc:`core/index`: The :mod:`core` module is
      responsible for the basic logic of fucrimodo. It further defines several
      abstracts that define the api for the different components of the multi
      stage algorithm.

    - **Custom Implementations** - :doc:`customs/index`: The :mod:`customs`
      module contains concrete implementations of the :mod:`core`. It currently implements a
      highly customizable genetic algorithm.

    - **Analyse Results** - :doc:`analysis/index`: The :mod:`analysis` module
      implements utilities to allow for quick analysis of results.

    - **Fucrimodo Lab** - :doc:`lab/index`: Human readable database to allow for
      reproducible experimentation.

    - **Command line interface** - :doc:`cli/index`: Interface to use and create
      the :mod:`fucrimodo_lab`.


This documentation aims to describes how classes and methods interact with each other and tries to motivate implementing your own algorithm ideas.

If you find errors or if you have problems understanding something, please open an issue in the repo: https://github.com/OHANAN1/fucrimodo/issues

.. code-block:: text

         .           ,
  .      .          /|
   .    .          /_/ ,
  .     .         /o \/|
 .     .          \<_/\|
  .    .           \ \ `
  .     .           \| Max
 .     /mlb          `

`What do you know about humans, Brunhilde? They spoil the sea.` - Fujimoto.
