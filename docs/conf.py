# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FUCrIMODo"
copyright = "2026, Louis Boehm"
author = "Louis Boehm, Martin Kuban"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # Optional, für automatisch generierte Zusammenfassungen
    "nbsphinx",  # For Jupyter Notebooks
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True  # Erzeugt automatisch Zusammenfassungen

# Führe Notebooks immer aus wenn sie gerendert werden
nbsphinx_execute = "auto"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme = "sphinx_rtd_theme"
hhtml_static_path = ["_static"]
html_css_files = ["custom.css"]
