.. FUCrIMODo documentation master file, created by
   sphinx-quickstart on Thu Sep 12 15:25:51 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=========
FUCrIMODo
=========

.. image:: _static/Fujimoto_legal.png
   :alt: Fujimoto from the Movie Ponyo by Studio Ghibli (source: https://www.ghibli.jp/works/ponyo/)
   :width: 200px
   :align: center

|

**F**\ ind **U**\ nknown **C**\ rystals by **I**\ nversion of **M**\ L **O**\ ptimized **D**\ escriptors

FUCrIMODo is a program that aims to retrieve the atomic structure from
a descriptor. Its current implementation allows to
This is done with the help of a novel multi-stage Genetic Algorithm (GA).
Fucrimodo has a modular design.


Information about GAs can be found at various sources. 
The DEAP library for example provides a very good
`documentation <https://deap.readthedocs.io/en/master/>`_ explaining 
concepts but also technical details of such algorithms.

Installation
============
The installation of FUCrIMODo is done via conda or pip.
Since conda is the recommended way to install the dependencies,
we will use it in the following example. Enter the directory of fucrimodo,
where the `environment.yml` file is located and run the following commands:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate fucrimodo-env

In this conda environment, `fucrimodo` is installed and added to the path as
a command line client. You can now run the program by typing `fucrimodo` in
the terminal to see the help message.

TODO: Add installation with virtual env

To fully customize fucrimodo refere to :doc:`development/index`

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   about
   getting_started/index
   tutorials/index
   fucrimodo/index
   development/index
   citation



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
