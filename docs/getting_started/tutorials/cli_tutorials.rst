
=============
CLI Tutorials
=============

This guide walks through the **fucrimodo** command-line interface (CLI). To install the library and CLI, see :doc:`../fucrimodo/getting_started/index`. For technical details about the CLI, see :doc:`../fucrimodo/cli/index`.

All commands and subcommands include built-in help, which you can access like this:

.. code-block:: bash

   fucrimodo --help
   fucrimodo run --help

The following tutorials are intended to be run in order.

.. contents:: CLI Tutorials
   :local:
   :depth: 1

======================
Create a Fucrimodo Lab
======================

To use the fucrimodo CLI, you first need to create a **fucrimodo lab**. A lab is a directory structure used to store runs, configs, and data. For more details, see :doc:`../fucrimodo/lab/index`.

A typical lab looks like this::

   fucrimodo_lab
   ├── configs
   │   ├── analysis
   │   ├── run
   │   └── utils
   ├── data
   │   └── raw
   └── README.md

Create a lab by running the ``init`` command. This creates a directory named ``fucrimodo_lab`` in the current working directory:

.. code-block:: bash

   fucrimodo init

Then enter the lab and inspect its contents:

.. code-block:: bash

   cd fucrimodo_lab
   tree

If ``tree`` is not installed, you can use ``ls -R`` or your file manager instead.

=====================
Generate Target Files
=====================

First, we need a target file containing the descriptor features we want to
invert.  The following command calculates the global SOAP descriptor features of
a structure (stored in a structure file: *.xyz) and stores them in a fucrimodo
target file (.json):

.. code-block:: bash

   fucrimodo utils \
       -a atoms_path=data/raw/simple-target.xyz \
       -a save_path=data/raw/simple-target.json

For details on the **fucrimodo utils** subcommand, see :doc:`../fucrimodo/cli/utils`.

=============
Perform a Run
=============

Next we can run the descriptor inversion with a custom run config.

Use the ``-c`` flag to specify a custom run config. If no config is provided,
the default run is performed. The default config can be found at
``configs/run/default.py``.

For all fucrimodo commands except ``init``, you can use ``-c`` to load a custom
config from a file. Later you will learn how to create such files.

The following command will perform a run on the selected config found at
``configs/run/test_run_config.py``.

.. code-block:: bash

   fucrimodo run \
       -v \
       -c configs/run/test_run_config.py \
       -s data/results/ \
       -n test_run \
       data/raw/simple-target.json

For details on the **fucrimodo run** subcommand, see :doc:`../fucrimodo/cli/run`.

The dir where results are stored is given by the specified save directory ("-s")
and name ("-n"). The results are therefore stored in ``data/results/test_run``::

   data/
   ├── raw
   │   └── ...
   └── results
       └── test_run  <-- results are stored here

Inspect the output:

.. code-block:: bash

   tree data/results/test_run

At minimum, the run produces the following files and directories. Depending on the config, additional files may also be generated::

   data/results/test_run
   ├── info.json
   ├── global_statistics.json
   ├── structures.db
   ├── stage_1
   ├── stage_2
   ├── stage_3
   └── ...

All output files are either human-readable or easy to process with common CLI tools such as ``jq``. See `Analyse With jq`_ for examples. Fucrimodo also ships with dedicated analysis tools, described in the next sections.

===================
Analyse Run Results
===================

For details on the **fucrimodo analyse** subcommand, see :doc:`../fucrimodo/cli/analyse`.

You can analyse a full run, a single stage, or multiple runs at once. Multi-run analysis is covered later in this tutorial.

Get an overview of a run:

.. code-block:: bash

   fucrimodo analyse run data/results/test_run/

Then select a row from the statistics table to inspect:

.. code-block:: bash

   fucrimodo analyse run -r 0 data/results/test_run/

This opens a plot for the selected statistic.

To analyse a single stage:

.. code-block:: bash

   fucrimodo analyse stage -r 0 data/results/test_run/stage_1/

(`That's the most boring goldfish I've seen in my entire life.`~Kumiko)

============
Target Files
============

Earlier in the tutorial we created a target file. Let us inspect it with ``jq``. If ``jq`` is not installed, open ``data/raw/simple-target.json`` in any text editor.

List the top-level keys:

.. code-block:: bash

   jq 'keys' data/raw/simple-target.json

Output::

   [
       "additional_notes",
       "descriptor_name",
       "features",
       "parameters"
   ]

Inspect the ``additional_notes`` field:

.. code-block:: bash

   jq '.additional_notes' data/raw/simple-target.json

Output::

   "Information:\nNumber of atoms: 1\nChemical formula: Fe\nCell volume: 27.0\nPBC: [ True  True  True]\n"

The ``Number of atoms`` entry controls how many atoms are used when the target is processed during a multi-stage run.

To create a target with a different number of atoms, use the ``utils`` command:

.. code-block:: bash

   fucrimodo utils \
       -c configs/utils/change_n_atoms.py \
       -a origin=data/raw/simple-target.json \
       -a out=data/raw/other-target.json \
       -a n_atoms=2

This creates a copy of the old target file but changes the ``Number of atoms``
entry to 2.  The new file is located at ``data/raw/other-target.json``.  You can
achive the same by manually editing the JSON, but this example demonstrates how
workflows can be automated with fucrimodo utilities.

==============
Create Configs
==============

Now let us create a custom config.

Run configs are located in ``configs/run``. Make a copy of an existing config:

.. code-block:: bash

   cp configs/run/test_run_config.py configs/run/custom_run_config.py

Open the new config in your preferred editor:

.. code-block:: bash

   $EDITOR configs/run/custom_run_config.py

Locate the ``main`` method and look around.
Next, find the line that contains ``global_rng = np.random.default_rng(42)`` (located after imports) and change the seed from ``42`` to ``43``.

The change is small, but it shows how configs control run behaviour. To understand configs in more depth, see :ref:`fucrimodo_as_library`. Continue with the remaining tutorials first, though.

=====================
Perform Multiple Runs
=====================

In many cases it is necessary to perform multiple runs at once. For example, if the correct number of atoms is unknown, you may need to run targets with different atom counts.

Earlier we created a target file with ``n_atoms = 2``. Let us run all targets in ``data/raw`` using a simple bash script. We will also use our new custom config.

Create ``scripts/run_all_targets.sh`` with the following content:

.. code-block:: bash

   path_to_target_files=data/raw
   results_path=data/results/multi-run

   mkdir -p "$results_path"

   for f in "$path_to_target_files"/*.json; do
       fucrimodo run \
           -p 4 \
           -c configs/run/custom_run_config.py \
           -s "$results_path" \
           "$f"
   done

Run the script. It will take a while.

You can monitor progress in another shell by tailing the stage logs:

.. code-block:: bash

   tail -f data/results/multi-run/*/stage_1/stage.log

You can do this for any run and any stage.

=====================
Analyse Multiple Runs
=====================

.. note::

   This section assumes you have completed all previous tutorials.

In the last section we ran two additional inversions. Their results are stored in ``data/results/multi-run``. To compare the runs, run:

.. code-block:: bash

   fucrimodo analyse multi_run data/results/multi-run

This produces two tables. The first is a multi-run overview showing, for each run, its name, description, number of stages, total number of generations, and total runtime.

The second table is a global statistics overview, showing the maximum and minimum values of each tracked statistic.

===================
Analyse with ASE DB
===================

.. code-block:: bash

   ase db data/results/test_run/structures.db

   ase db data/results/test_run/structures.db -w

[TODO: MISSING]

===============
Analyse With jq
===============

[TODO: MISSING]
