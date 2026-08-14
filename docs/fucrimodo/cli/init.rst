CLI Init
========

The ``init`` CLI generates the ``fucrimodo_lab`` directory with default configs.
For more information about the ``fucrimodo_lab`` please look at
:ref:`fucrimodo_lab_docs`

The lab template is copied from package data to the location specified with
``--save_dir``; if not given, the current working directory is used.  By
default, the user is asked to confirm before creating the directory.

---------
Arguments
---------

.. code-block:: text

    fucrimodo init [OPTIONS]

-------
Options
-------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``-s, --save_dir PATH``
     - Directory where the ``fucrimodo_lab`` directory should be
       created.  Must exist.  Defaults to the current working directory.
   * - ``-y, --yes``
     - Create the directory without asking for confirmation.
   * - ``-v, --verbose``
     - Enable verbose output.

-------
Example
-------

.. code-block:: bash

    fucrimodo init -s ./my-projects

More examples can be found in :ref:`cli-tutorials`.

------
Runner
------

.. autoclass:: fucrimodo.cli.init.Runner
   :members:
   :undoc-members:
   :show-inheritance:
