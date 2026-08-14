========
CLI Init
========

The ``init`` CLI generates a new ``fucrimodo_lab`` directory from the
built-in template. The lab contains the default configuration files
needed to run, store, and analyse multi-stage optimization experiments.


Command reference
-----------------

.. code-block:: text

    fucrimodo init [OPTIONS]

Options
~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``-s, --save_dir PATH``
     - Directory where the ``fucrimodo_lab`` directory is created.
       Defaults to the current working directory.
   * - ``-y, --yes``
     - Create the directory without asking for confirmation.
   * - ``-v, --verbose``
     - Enable verbose output.

Behaviour
~~~~~~~~~

The command copies the ``lab_template`` package data into a new
``fucrimodo_lab`` directory inside ``save_dir``.  If the directory
already exists, the command aborts with an error.  Unless ``--yes`` is
given, the user is asked to confirm before anything is created.

Example
~~~~~~~

.. code-block:: bash

    fucrimodo init -s ./my_projects -y
