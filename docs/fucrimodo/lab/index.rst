.. _fucrimodo_lab_docs:

=============
Fucrimodo Lab
=============

``fucrimodo_lab`` provides a lightweight way to run, organize, and reproduce
experiments with the MultiStageSearch algorithm — across configurations that
range from small parameter tweaks to fundamentally different setups. To make
experiments reproducible ``fucrimodo_lab`` suggests some best paractices.
These should impose minimal restrictions and keep data in a format
that's easy to read with common CLI or Python tools.

A related project, `sacred <https://github.com/IDSIA/sacred>`_, offers
structured reproducibility for experiment data (though not for run/analysis
configs). It may be integrated into ``fucrimodo_lab`` in the future.

(Maybe ``fucrimodo_lab`` could eventually become a standalone package for general
computational experiments, because I invested a lot of time in creating the
concept. But if there are projects out there that do the same thing, please tell
me about them!)


-----
Setup
-----

To set up the fucrimodo lab please refer to :ref:`cli-tutorials`.

---------
Structure
---------

::

   fucrimodo_lab
   ├── data
   ├── configs
   └── scripts

~~~~
Data
~~~~

::

   fucrimodo_lab/data/
   └── raw
       ├── example-target.xyz
       └── test-target.xyz

Sample data for benchmarks and tests. Store databases and target files here,
and place results under ``data/results/`` (use subdirectories to categorize
runs). If keeping large data local would clutter the repo, store it elsewhere
on the machine and pass the path explicitly during a run.

~~~~~~~
Configs
~~~~~~~

Configuration files used with the fucrimodo CLI — see the
:ref:`CLI tutorial <cli-tutorials>` and :ref:`CLI documentation
<cli-documentation>` for usage examples.

Each CLI subcommand has its own directory::

    fucrimodo_lab/configs
    ├── run
    ├── analyse
    └── utils

``analyse`` further splits based on the analysis objects::

    fucrimodo_lab/configs/analyse
    ├── multi_run
    ├── run
    └── stage

Each directory initially includes a ``default.py``, used automatically when no
config is specified — copy and adapt it to create your own configs. Every config
file must define a :meth:`main` method, with parameters depending on the
subcommand:

**run**

.. list-table::
   :header-rows: 1
   :widths: 30 30 20

   * - Parameter
     - Type
     - Default
   * - ``name``
     - ``str | None``
     - –
   * - ``save_dir``
     - ``str``
     - –
   * - ``target_file_path``
     - ``str``
     - –
   * - ``n_parallel``
     - ``int``
     - –
   * - ``verbose``
     - ``bool``
     - –

**analyse**

.. list-table::
   :header-rows: 1
   :widths: 30 30 20

   * - Parameter
     - Type
     - Default
   * - ``dir_path``
     - ``str``
     - –
   * - ``row``
     - ``int | None``
     - ``None``
   * - ``save_dir``
     - ``str | None``
     - ``None``
   * - ``show``
     - ``bool``
     - ``False``
   * - ``verbose``
     - ``bool``
     - ``False``

**utils**

.. list-table::
   :widths: 100

   * - Fully flexible, i.e. accepts any named variables, as long as they're
       declared in :meth:`main` and passed via ``-a``::

          fucrimodo utils -c path/to/conf.py -a some_var=value

       In this chase the :meth:`main` function in the conf.py file must accept
       ``some_var`` as input.



Run ``--help`` on any subcommand, or inspect the relevant ``default.py``, for
full parameter details.

~~~~~~~
Scripts
~~~~~~~

::

    fucrimodo_lab/scripts
    ├── perform_test_run.sh
    └── run_slurm_array.sh

``perform_test_run.sh`` mirrors the steps in the `tutorial` to verify your
setup with a simple descriptor inversion run - execute it from the lab root
with ``scripts/perform_test_run.sh``. It should finish in under a minute.

``run_slurm_array.sh`` is a commented, customizable script for deploying
fucrimodo on a Slurm cluster via job arrays rather than single-node
parallelism. Genetic algorithms are iterative, so parallelizing within one
run offers little speedup (data must repeatedly be merged and re-split).
But adjust if needed. This is only an example.
