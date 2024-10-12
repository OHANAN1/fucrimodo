.. FUCrIMODo documentation master file, created by
   sphinx-quickstart on Thu Sep 12 15:25:51 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=========
FUCrIMODo
=========

.. image:: _static/Fujimoto.jpg
   :alt: Fujimoto from the Movie Ponyo by Studio Ghibli
   :width: 200px
   :align: center

FUCrIMODo is a program that aims to retrieve the crystal structure from
a SOAP descriptor.
This is done with the help of a novel multi-stage Genetic Algorithm (GA).

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

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   modules/core/index
   modules/analysis/index
   modules/customs/index
   modules/cli/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
