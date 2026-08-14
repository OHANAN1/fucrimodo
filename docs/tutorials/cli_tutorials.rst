.. _cli-tutorials:

=============
CLI Tutorials
=============

To install the fucrimodo library with the fucrimodo cli please refere to the
document :doc:`../fucrimodo/getting_started/index`. For technical details about
the cli please refer to the :doc:`../fucrimodo/cli/index`.

All commands and subcommands come with help included. Help can be accessed like this::

  fucrimodo --help

  fucrimodo run --help


The following tutorials should be run one after the other.

.. contents:: Cli Tutorials
    :local:
    :depth: 1

----------------------
Create A Fucrimodo Lab
----------------------

To properly use the fucrimodo cli you need to create a ``fucrimodo_lab``. A ``fucrimodo_lab`` is a directory structure to store runs please read more about it in :doc:`../fucrimodo/lab/index`.
::

    fucrimodo_lab
    ├── configs
    │   ├── analysis
    │   ├── run
    │   └── utils
    ├── data
    │   └── raw
    └── README.md


A fucrimodo lab can be easily created with the cli. Just use the ``init`` command to create a directory called ``fucrimodo_lab`` in the current working dictionary.
.. code-block:: bash

  fucrimodo init


Now you can enter the fucrimodo_lab and look around
.. code-block:: bash

   cd fucrimodo_lab
   tree

Note: `tree` is optional. If not present on your system just look around with ls.

---------------------
Generate Target Files
---------------------

More info at: :doc:`..fucrimodo.cli.utils`.

.. code-block:: bash

   fucrimodo utils \
       -a atoms_path=data/raw/test-target.xyz \
       -a save_path=data/raw/test-target.json

-------------
Perform a Run
-------------

More info at: :doc:`..fucrimodo.cli.run`.

Set a custom config with -c. if no config flag is given then the default run is performt. Its script can be found in ``configs/run/test_run_config.py``
For all fucrimodo commands (appart from ``init``) there is the option -c with custom config. With this the manual config is set. To sea how custom configs work check out `Use Config Files`_.

.. code-block:: bash

   fucrimodo run \
       -v \
       -c configs/run/test_run_config.py \
       -s data/results/ \
       -n test_run \
       data/raw/test-target.json

results of the run are stored in "data/results/test_run".

::

    data/
    ├── raw
    │   └── ...
    └── results
        └── test_run  <== Results were stored here

You can look around there:

.. code-block:: bash

    tree data/results/test_run

This will generate minimum the files and depending on the scripts some additional things.

::

   data/results/test_run
    ├── info.json
    ├── global_statistics.json
    ├── structures.db
    ├── stage_1
    ├── stage_2
    ├── stage_3
    └── ...

All files should be easily human readible, or usable together with common
cli-tools like jq.  More information can be found `Analyse With jq`__.  However,
fucrimodo ships with analysis tools to perform the analysis. This will be
explained in the following.



-------------------
Analyse Run Results
-------------------

More info at: :doc:`..fucrimodo.cli.analyse`.

You can either analyse a run, a stage of a run. Also there is the option to
analyse multiple runs at once. This will be discussed later, tho.

Get an overview
.. code-block:: bash

   fucrimodo analyse run data/results/test_run/

Choose the row of the statistic you want to analyse:

.. code-block:: bash

   fucrimodo analyse run -r 0 data/results/test_run/


This opens a plot you can look at.
(`That's the most boring goldfish I've ever seen in my entire life.`~Kumiko)

To analyse stages:
.. code-block:: bash

    fucrimodo analyse stage -r 0 data/results/test_run/stage_1/

--------------
Create Configs
--------------

Now we can try to run a custom config.

We can find run configs in ``configs/run``

Please make a copy of one of the existing configs:

.. code-block:: bash

    cp configs/run/test_run_config.py configs/run/custom_run_config.py

Now you can open the run config with your editor of choice (Nvim, Emacs, VS Code, ...):

.. code-block:: bash

    edit configs/run/custom_run_config.py

Now look. Look for the :meth:``main`` method. For more information on how the individual components of fucrimodo work look at :doc:`../../tutorials/fucrimodo_as_a_library.ipynb`.

Next please locate the entry ``global_rng = np.random.default_rng(42)`` and change the seed ``42`` to ``43``

.. code-block:: bash

    fucrimodo run -p 8 -c configs/run/custom_run_config.py -s data/results/ -n custom_config_run data/raw/test-target.json


Congrats, you ran your first custom inversion. Now please check the results at ``data/results/custom_config_run`` and analyse them like explained above. E.g.:

.. code-block:: bash

    fucrimodo analyse run -r 0 data/results/custom_config_run


For more info about the concept of configs in the ``fucrimodo_lab`` look at :doc:`../fucrimodo/lab/index`.

---------------------
Analyse Multiple Runs
---------------------

This is now

.. code-block::

    fucrimodo analyse multi_run data/results/


---------------------
Perform Multiple Runs
---------------------

To test how to analyse

Run on different files by using a bash for loop.
Create a bash script to run multiple files:

Lets say there exists another target file we want to invert.


[TODO: MISSING]

---------------
Analyse With jq
---------------

[TODO: MISSING]
