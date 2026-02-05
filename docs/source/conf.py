# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "AeroSim Docs"
copyright = "2021-2026, AeroSim"
author = "AeroSim"


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
    "shibuya",
]

source_suffix = [".rst", ".md"]
bibtex_bibfiles = ["_refs/refs.bib"]

# Include TODOs (check https://www.sphinx-doc.org/en/master/usage/extensions/todo.html)
todo_include_todos = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"

html_favicon = "_static/favicon.svg"
html_title = "AeroSim CFDmod"
html_context = {
    "plausible_script": """
    <!-- Plausible -->
    <script defer data-domain="aerosim.io" src="https://plausible.io/js/script.file-downloads.hash.outbound-links.pageview-props.tagged-events.js"></script>
    <script>window.plausible = window.plausible || function() { (window.plausible.q = window.plausible.q || []).push(arguments) }</script>
    """
}

# Theme options
html_theme_options = {}
html_static_path = ["_static"]
html_css_files = ["custom.css"]


html_theme_options = {
    "nav_links": [
        {"title": "Docs", "url": "https://docs.aerosim.io"},
        {"title": "Nassu", "url": "https://docs.aerosim.io/nassu"},
        {"title": "CFDMod", "url": "https://docs.aerosim.io/cfdmod"},
        {"title": "AeroSim", "url": "https://docs.aerosim.io/aerosim"},
    ],
    "nav_links_align": "center",
    "light_logo": "_static/img/logo_display.png",
    "dark_logo": "_static/img/white_logo.png",
    "logo_target": "https://aerosim.io",
}
