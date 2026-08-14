
.. _cli-documentation:

=============
Fucrimodo CLI
=============

The command-line interface (CLI) is the main way to interact with FUCrIMODo.
This section documents the technical details of each CLI module.
For hands-on tutorials and usage examples, see :doc:`../../tutorials/cli_tutorials`.


::

   $: fucrimodo --help
   Usage: fucrimodo [OPTIONS] COMMAND [ARGS]...

     Fucrimodo command line tool.

     Fucrimodo helps configure, run, store, and analyse reproducible multi-stage
     optimization experiments.

     Available commands:

       init      Create a new fucrimodo_lab directory with default configs.
       run       Run an optimization experiment or stage.
       analyse   Analyse data collected during a run or stage.
       utils     Utility commands for managing fucrimodo projects.

     To start please read the tutorials in the docs or if you feel brave just
     create a new fucrimodo_lab with 'fucrimodo init'.

   Options:
     -V, --version  Show the version and exit.
     --help         Show this message and exit.

   Commands:
     analyse  Analyse run or stage data.
     init     Generate a fucrimodo_lab directory with default configs.
     run      Perform a multi-stage search run on the provided input file.
     utils    Execute a utility configuration script.

     Run 'fucrimodo COMMAND --help' for more information on a specific command.

     Example:
       fucrimodo init --save_dir ./my_experiments
       fucrimodo run -s 'data' -n 'run_01' ./data/raw/example_target_file.json
       fucrimodo analyse run ./data/run_01




The CLI is organised into the following submodules:

.. toctree::
    :maxdepth: 1

    init
    run
    analyse
    utils
