=====================
Reproduce Publication
=====================

.. note::

   These tutorials are advanced. Please familiarize yourself first with the
   tutorials :ref:`cli-tutorials` and :ref:`fucrimodo_as_library`.


The following tutorials will explain how to recreate some of the main results of
the :ref:`publication<citation>` that introduced the library. Currently only a few
tutorial are provided. The rest is coming soon.

.. _three_example_inversions:

========================
Three Example Inversions
========================

To invert the three examples Ag2Se4Y2, Si2O4 and FeCl2 you first have to create target files.
All three targets are available in the default fucrimodo lab. With this script it can be easily done.
Please run this inside the fucrimodo lab.

.. code-block:: bash

    for f in data/raw/publication-targets/* ; do
        fucrimodo utils \
            -a atoms_path=f \
            -a save_path="${f%.*}".json
    done

Next the runs can be performed. We use the default config, assuming it is unchanged.

.. code-block:: bash

   path_to_target_files=data/raw
   results_path=data/results/multi-run

   mkdir -p "$results_path"

   for f in "$path_to_target_files"/*.json; do
       fucrimodo run \
           -p 4 \
           -s "$results_path" \
           "$f"
   done

=======================
Analyse 1000 Inversions
=======================

To analyse the data generated for the big run with 1000 inversions from the
:ref:`publication<citation>`, please first download the generated data from
`Zenodo <https://doi.org/10.5281/zenodo.22031213>`__.  After downloading, unzip
the file ``1000-targets-mp-20-max-10-atoms.zip`` and place the folder inside the
fucrimodo lab at ``data/results/``.  The analysis config script is shipped  with
the default fucrimodo lab and can be found at
``config/analyse/multi_run/matching_per_n_atoms.py``.  This script requires the
additional dependency ``seaborn``. Please install it in the environment with
``pip install seaborn``. Now you can perform the analysis with:

.. code-block:: bash

   fucrimodo analyse multi_run \
       -c configs/analyse/multi_run/matching_per_n_atoms.py \
       data/results/100-targets-mp-20-max-10-atoms-multi-seed

Note that the runs ``None_id_21``, ``None_id_368`` and ``None_id_859`` are the
inversions of the three example structures from the publication. Please refer to
the :ref:`cli-tutorials` to analyse the runs individually.


=======================
Analyse Robustness Test
=======================

The robustness analysis from the :ref:`publication<citation>` can be performed
in the same way as in the section before. This is because all 10
structures have a different number of atoms.
