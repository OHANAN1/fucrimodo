.. _installation:

============
Installation
============

The latest version of ``fucrimodo`` requires **Python >= 3.12**. If this Python
version is not available on your system and you do not have admin rights, please
refer to `Using conda`_ or `Using uv`_.

----------------
Using venv + pip
----------------

Like every Python project, it is recommended to use a virtual environment. More
information can be found in the official `Python Documentation
<https://docs.python.org/3/library/venv.html>`_. If you prefer to have the
environment be managed automatically, refer to `Using conda`_ or `Using uv`_.

Create a virtual environment with Python 3.12 or higher and activate it:

.. code-block:: bash

   python3.12 -m venv .venv
   source .venv/bin/activate

Now, install ``fucrimodo`` from PyPI:

.. code-block:: bash

   pip install fucrimodo

Verify that the ``fucrimodo`` command is available:

.. code-block:: bash

   fucrimodo --help

-----------
Using conda
-----------

The package is available on `conda-forge <https://conda-forge.org>`_.

If you do not already have a conda environment, create and activate one first:

.. code-block:: bash

   conda create -n fucrimodo-env python=3.12
   conda activate fucrimodo-env

Install from conda-forge:

.. code-block:: bash

   conda install -c conda-forge fucrimodo

Verify that the ``fucrimodo`` command is available:

.. code-block:: bash

   fucrimodo --help

--------
Using uv
--------

The program can also be installed with the package manager `uv
<https://docs.astral.sh/uv/>`__.

Create a virtual environment:

.. code-block:: bash

   uv venv --python 3.12

This creates a virtual environment in ``.venv``. Activate it and install the
package from PyPI:

.. code-block:: bash

   source .venv/bin/activate
   uv pip install fucrimodo

If you already have a virtual environment activated, you can skip the
``uv venv`` step.

Verify that the ``fucrimodo`` command is available:

.. code-block:: bash

   fucrimodo --help

-----------
From Source
-----------

To install the latest development version from source, clone the repository from
`GitHub <https://github.com/OHANAN1/fucrimodo>`__:

.. code-block:: bash

   git clone https://github.com/OHANAN1/fucrimodo
   cd fucrimodo

Make sure you have an active virtual environment (see `Using venv + pip`_,
`Using conda`_, or `Using uv`_), then install with pip:

.. code-block:: bash

   pip install .

For an editable development install, use:

.. code-block:: bash

   pip install -e .

Alternatively, set up the environment automatically with conda:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate fucrimodo-env

Or with uv:

.. code-block:: bash

   uv sync

---------------------
Test the installation
---------------------

If you installed from source, make sure you are in the repository root and have
your virtual environment activated. Then install the test dependencies and run the
test suite:

.. code-block:: bash

   pip install ".[test]"

.. code-block:: bash

   pytest

The full test suite takes around 2 minutes to complete.

If you installed ``fucrimodo`` from PyPI, conda-forge, or uv, the test files are
not included in the package. Please refer to `From Source`_.
