=====================
Reproduce Publication
=====================

.. note::

   These tutorials are advanced. Please familiarize yourself first with the
   tutorials :ref:`cli-tutorials` and :ref:`fucrimodo_as_library`.


The following tutorials will explain how to recreate some of the main results of
the :ref:`publication<citation>` that introduced the library. Currently only the
Tutorial for :ref:`three_example_inversions` is implemented. The rest is coming
soon.

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

===============
1000 Inversions
===============

[MISSING]

===============
Robustness Test
===============

[MISSING]
