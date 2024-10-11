# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FUCrIMODo'
copyright = '2024, Louis Boehm'
author = 'Louis Boehm, Martin Kuban'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',  # Optional, für automatisch generierte Zusammenfassungen
    'nbsphinx', # For Jupyter Notebooks
    'sphinxcontrib.tikz' # To render tikz code
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True  # Erzeugt automatisch Zusammenfassungen

# Führe Notebooks immer aus wenn sie gerendert werden
nbsphinx_execute = 'always'

# Configuriere Latex
tikz_tikzlibraries = "arrows, shapes.geometric, positioning, shapes.multipart, arrows.meta, fit"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
