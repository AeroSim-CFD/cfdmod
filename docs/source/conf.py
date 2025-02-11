# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CFD Modules"
copyright = "2023, Waine Oliveira Jr, Pablo Penas"
author = "Waine Oliveira Jr, Pablo Penas"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ["_templates"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinxcontrib.bibtex",
    "myst_parser",
    "nbsphinx",
]

source_suffix = [".rst", ".md"]
bibtex_bibfiles = ["_refs/refs.bib"]

# Include TODOs (check https://www.sphinx-doc.org/en/master/usage/extensions/todo.html)
todo_include_todos = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_title = "CFD Mods"

html_static_path = ["_static"]
