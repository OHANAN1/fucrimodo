CLI Run
===========

The ``run`` CLI performs a multi-stage search on a provided input file.

The search is configured through a configuration script loaded via
:py:class:`~fucrimodo.utils.import_helper.ConfigScript`.  Results are
saved to a directory that can be specified with ``--save_dir``; if not
given, the current working directory is used.

Command reference
-----------------

.. code-block:: text

    fucrimodo run [OPTIONS] INPUT_FILE

Arguments
~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Argument
     - Description
   * - ``INPUT_FILE``
     - Path to the input/target file to process.  Must exist.

Options
~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``-c, --config PATH``
     - Path to the run configuration script.  Defaults to
       ``configs/run/default.py``.
   * - ``-s, --save_dir PATH``
     - Directory where the run outputs are saved.  Created if it does
       not exist.  Defaults to the current working directory.
   * - ``-n, --name TEXT``
     - Name of the run.
   * - ``-p, --parallel INTEGER RANGE``
     - Number of parallel workers/processes to use.  Must be at least
       1.  Default: ``1``.
   * - ``-v, --verbose``
     - Enable verbose output.

Example
~~~~~~~

.. code-block:: bash

    fucrimodo run targets.csv -c configs/run/default.py -s results/run_1 -n my_run -p 4 -v


Runner
~~~~~~

.. autoclass:: fucrimodo.cli.run.Runner
   :members:
   :undoc-members:
   :show-inheritance:
