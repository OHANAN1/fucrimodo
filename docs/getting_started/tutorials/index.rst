.. _tutorials:

=========
Tutorials
=========

Before you begin, make sure you have read the :ref:`getting-started` guide and
:ref:`installed fucrimodo <installation>`.

The tutorials below introduce you to using fucrimodo:

- :ref:`cli-tutorials`  - run your first descriptor inversion experiment.
- :ref:`fucrimodo_as_library` - use fucrimodo as a library inside a jupyter
  notebook to compose, execute and analyse your own multi-stage search
  algorithms.

The advanced tutorials cover more specialised topics:

- :ref:`run_on_slurm` - run experiments on a SLURM cluster.
- :ref:`reproduce_publication` - reproduce the results of the
  :ref:`publication<citation>` in which the library was introduced.


.. _cli-tutorials:

-----------------
Fucrimodo As CLI
-----------------

.. toctree::
   :maxdepth: 2

   cli_tutorials


.. _fucrimodo_as_library:

--------------------
Fucrimodo As Library
--------------------

.. toctree::
    :maxdepth: 1

    fucrimodo_as_library


.. _run_on_slurm:

------------
Run on Slurm
------------

.. note::
   The currently provided Slurm script is only experimental and it is not
   guaranteed to work properly.

`Slurm <https://slurm.schedmd.com/overview.html>`__ is a widely used workload
manager for HCP cluster.  The Slurm script can be found in
``fucrimodo_lab/scripts/run_slurm_array.sh``.  Please look at the comments
inside the file to set it up or create your own (and potentially better) slurm
scripts. This tutorial will be extended once the Slurm script is properly
tested.

[MISSING]

.. _reproduce_publication:

---------------------
Reproduce Publication
---------------------

.. toctree::
    :maxdepth: 1

    reproduce_publication
