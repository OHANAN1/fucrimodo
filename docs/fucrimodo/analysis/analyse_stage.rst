==============
Analyse Stages
==============

.. currentmodule:: fucrimodo.analysis.analyse_stage

The data can be loaded with the :class:`StageData` class.
::

    from fucrimodo.analysis.analyse_run import StageData

    stage_data = StageData('path/to/stage/dir')

Tutorials to explain analysis and visualization of the loaded data can be found here:
[TODO: Add missing tutorials].

To enable analysis of custom stages please provide a suitable data loader. The
implementation used to analyse the :class:`GAStage` can be found in
:mod:`fucrimodo.customs.ga_stage.analysis` and can be adapted to any custom
stage.

-----------
Data Loader
-----------

.. autoclass:: fucrimodo.analysis.stage_analysis.StageData
   :members:
   :undoc-members:
   :show-inheritance:

---------
Utilities
---------

.. automethod:: fucrimodo.analysis.stage_analysis.get_modification_overview

.. automethod:: fucrimodo.analysis.stage_analysis.get_stage_overview
