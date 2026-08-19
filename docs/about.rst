.. _about:

=====
About
=====

FUCrIMODo is a Python program for retrieving atomic structures from materials
science descriptors via a novel multi-stage search algorithm. It provides a
command-line interface (CLI) as well as a Python library, which can also be used
to configure the CLI program. FUCrIMODo was implemented with the following
design goals in mind:

**Modular design**
    Every aspect of FUCrIMODo is fully modular, and all configurations can be
    replaced. While this makes implementation and maintenance more difficult,
    we believe this flexibility is essential for developing creative new
    algorithms.

**Reproducibility**
    FUCrIMODo aims to make all results reproducible, so that even when parts
    of the program are reworked, previous results can still be analysed and
    repeated. Some design aspects are inspired by the `sacred library
    <https://sacred.readthedocs.io/en/stable/index.html>`__.

**Python friendly**
    To simplify the implementation of new algorithms and configurations, all
    methods and classes have full type hints. This enables fast development
    and makes the main API quick to learn. We also rely on libraries commonly
    used in data science, such as pandas and NumPy, so that data structures
    can be easily handled.
