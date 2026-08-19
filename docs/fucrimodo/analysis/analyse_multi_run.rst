=====================
Analyse Multiple Runs
=====================

.. currentmodule:: fucrimodo.analysis.analyse_run

If you want to compare multiple runs with each other or get statistics from all
runs they can be structured with the :class:`MultiRunData`. Please provide a
path to a directory that contains multiple run directories. Then the data can be
loaded with the :class:`MultiRunData` class.
::

    from fucrimodo.analysis.run_results import MultiRunData

    multi_run_data = MultiRunData('path/to/run/dir')

Tutorials to explain analysis and visualization of the loaded data can be found here:
:ref:`fucrimodo_as_library`.

-----------
Data Loader
-----------

.. autoclass:: fucrimodo.analysis.multi_run_analysis.MultiRunData
   :members:
   :undoc-members:
   :show-inheritance:

Info about run analysis can be found here: :doc:`analyse_run`.

---------
Utilities
---------

.. automethod:: fucrimodo.analysis.multi_run_analysis.get_all_global_statistics_overview

.. automethod:: fucrimodo.analysis.multi_run_analysis.get_multi_run_overview

