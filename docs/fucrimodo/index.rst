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

    - **Basic Framework**
        The :doc:`core/index` module is responsible for the
        basic logic of fucrimodo. It further defines several abstracts that define
        the api for the different components of the multi stage algorithm.
    - **Custom Implementations**
        The :doc:`customs/index` module contains
        concrete implementations of the :mod:`core`. It currently implements a
        highly customizable genetic algorithm.
    - **Analyse Results**
        The :doc:`analysis/index` module implements
        utilities to allow for quick analysis of results.
    - **Fucrimodo Lab**
        The :doc:`lab/index` is a human readable database that
        allows for reproducible experimentation.
    - **Command line interface**
        The :doc:`cli/index` provides an interface to
        create and interact with the :mod:`fucrimodo_lab`.




This documentation aims to describes how classes and methods interact with each other and tries to motivate implementing your own algorithm ideas.

If you find errors or if you have problems understanding something, please open an issue in the repo: https://github.com/OHANAN1/fucrimodo/issues .

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
