Analyse run Dokumentation
=========================

The analysis objects are used to analyse the data that was loaded into the 
Results objects (see :doc:`results_class`).
They can be initialized either with a Results object or with a
path to a results file. 
For the :class:`AnalyseRun` class this can for example look something like 
this::

    from fucrimodo.analysis.analyse_run.run_analysis import AnalyseRun
    from fucrimodo.analysis.analyse_run.run_results import RunResults

    results = RunResults('path/to/results')
    analysis = AnalyseRun(results)

    # or

    analysis = AnalyseRun('path/to/results')

The analysis objects can then be used to get all kinds of information about the
run, the stages, the results and the data. 
To do this an analysis key needs to be selected for every method.
These keys are the same as the keys of the statistics that where collected
during the run.
For example the key reference_similarity could be used to analyse how the
reference similarity changed over the course of the run.
Some methods also require a `value_type` to be selected, normally this
is one of the following: 'mean', 'std', 'min', 'max'.
It depends on what statistic was tracked, but normally these are the available
value types.

.. currentmodule:: fucrimodo.analysis.run_analysis

.. autosummary::
   :recursive:

   RunData

.. automodule:: fucrimodo.analysis.run_analysis
   :members:
   :undoc-members:
   :show-inheritance:


.. currentmodule:: fucrimodo.analysis.stage_analysis

.. autosummary::
   :recursive:

   StageData

.. automodule:: fucrimodo.analysis.stage_analysis
   :members:
   :undoc-members:
   :show-inheritance:
