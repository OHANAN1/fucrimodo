=========
CLI Utils
=========

The ``utils`` CLI provides a easy way to run small utility scripts
defined by configuration files.

Each utility is executed through a configuration script that is loaded through
:py:class:`~fucrimodo.utils.import_helper.ConfigScript`.  Parameters are passed
as ``key=value`` pairs from the command line and forwarded to the configuration
script as keyword arguments. The config script then has to accept these named
parameters and perform the utility. An example config script can be found in the
default fucrimodo lab at ``configs/utils/create_target_file.py``.

---------
Arguments
---------

.. code-block:: text

    fucrimodo utils [OPTIONS]

-------
Options
-------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Option
     - Description
   * - ``-c, --config PATH``
     - Path to the utility configuration script.  Utility configs are
       expected under ``configs/utils/`` inside the ``fucrimodo_lab``
       directory.  See the ``README.md`` for more information.
   * - ``-a, --arg KEY=VALUE``
     - Argument passed to the utility configuration script as a keyword
       argument.  Can be given multiple times.
   * - ``-v, --verbose``
     - Enable verbose output.

-------
Example
-------

.. code-block:: bash

   fucrimodo utils \
       -a atoms_path=data/raw/test-target.xyz \
       -a save_path=data/raw/test-target.json

More examples can be found in :ref:`cli-tutorials`.

------
Runner
------

.. autoclass:: fucrimodo.cli.utils.Runner
   :members:
   :undoc-members:
   :show-inheritance:
