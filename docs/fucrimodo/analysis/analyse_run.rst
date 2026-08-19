============
Analyse Runs
============

.. currentmodule:: fucrimodo.analysis.analyse_run

The data can be loaded with the :class:`RunData` class.
::

    from fucrimodo.analysis.run_results import RunData

    run_data = RunData('path/to/run/dir')

Tutorials to explain analysis and visualization of the loaded data can be found
here: :ref:`fucrimodo_as_library`.

-----------
Data Loader
-----------

.. autoclass:: fucrimodo.analysis.run_analysis.RunData
   :members:
   :undoc-members:
   :show-inheritance:

Info about stage analysis can be found here: :doc:`analyse_stage`.

---------
Utilities
---------

.. automethod:: fucrimodo.analysis.run_analysis.get_best_individual

.. automethod:: fucrimodo.analysis.run_analysis.get_run_overview

