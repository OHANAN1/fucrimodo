===========
CLI Analyse
===========

The ``analyse`` CLI is used to analyse data collected during a ``run`` or ``stage``.

The analysis is configured through a configuration script loaded via
:py:class:`~fucrimodo.utils.import_helper.ConfigScript`.  If no custom
configuration is provided, the default configuration for the selected
analysis object is loaded from ``configs/analyse/<ANALYSIS_OBJECT>/default.py``.
Results are saved to a directory specified with ``--save_dir``; if not
given, results are displayed instead.

---------
Arguments
---------

.. code-block:: text

    fucrimodo analyse [OPTIONS] ANALYSIS_OBJECT DIR_PATH

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Argument
     - Description
   * - ``ANALYSIS_OBJECT``
     - Type of object to analyse.  Must be one of: ``run``, ``stage``,
       ``multi_run``.
   * - ``DIR_PATH``
     - Directory where the run or stage results are saved.  Must exist.

-------
Options
-------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``-v, --verbose``
     - Enable verbose output.
   * - ``-s, --save_dir PATH``
     - Directory where the analysis results are saved.  Must exist.  If
       omitted, results are displayed instead.
   * - ``-r, --row INTEGER``
     - Row index to analyse.  If not provided, all rows are analysed.
   * - ``-c, --config PATH``
     - Path to a custom analysis configuration script. Defaults to
       ``configs/analyse/<object>/default.py``.

-------
Example
-------

.. code-block:: bash

    fucrimodo analyse stage data/results/example-run/stage_1

More examples can be found in :ref:`cli-tutorials`.

------
Runner
------

.. autoclass:: fucrimodo.cli.analyse.Runner
   :members:
   :undoc-members:
   :show-inheritance:
