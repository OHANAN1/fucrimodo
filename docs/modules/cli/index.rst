=================
CLI Dokumentation
=================

The command line interface is the main way to interact with FUCrIMODo. 
To use it simply run::

    python fucrimodo

This will print the help message which will explain further details.

One of the main use chases of the CLI is to perform an inversion on a 
descriptor specified in an input file 
(MISSING: LINK TO INFO ABOUT INPUT FILE)::

    python fucrimodo run PATH_TO_INPUT_FILE

To analyse the results of a run that where saved to a specific results 
directory (MISSING: INFO ABOUT RESULTS DIR) a notebook can be created::

    python fucrimodo analyse notebook PATH_TO_RESULTS_DIR
    
The generation of plots, figures, tables or other results data can also be
done without the creation of notebooks. Just use::

    python fucrimodo analyse run PATH_TO_RESULTS_DIR

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   run
   analyse
