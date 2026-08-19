.. _cli-run-documentation:

=======
CLI Run
=======

The ``run`` CLI performs a multi-stage search on a provided input file.

The search is configured through a configuration script loaded via
:py:class:`~fucrimodo.utils.import_helper.ConfigScript`. An example config
script can be found in the default fucrimodo lab at ``configs/run/default.py``.
Results are saved to a directory that can be specified with ``--save_dir``; if
not given, the current working directory is used.

---------
Arguments
---------

.. code-block:: text

    fucrimodo run [OPTIONS] INPUT_FILE


.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Argument
     - Description
   * - ``INPUT_FILE``
     - Path to the input/target file to process.  Must exist.

-------
Options
-------

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

-------
Example
-------

.. code-block:: bash

    fucrimodo run \
        -s data/results/ \
        data/raw/test-target.json

More examples can be found in :ref:`cli-tutorials`.

------
Runner
------

.. autoclass:: fucrimodo.cli.run.Runner
   :members:
   :undoc-members:
   :show-inheritance:
