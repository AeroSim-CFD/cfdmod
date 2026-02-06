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

# -- nbsphinx configuration --------------------------------------------------
# Never execute notebooks during build
nbsphinx_execute = "never"

# Configure nbsphinx for dark mode support
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: html

    <style>
        /* Estilos base para notebooks */
        div.nbinput, div.nboutput {
            margin: 1em 0;
            border-radius: 4px;
        }
        
        div.nbinput .prompt, div.nboutput .prompt {
            min-width: 11ex;
            padding: 0.4em;
            font-family: monospace;
        }
        
        /* Dark mode support for notebooks */
        html[data-theme="dark"] div.nbinput,
        html[data-theme="dark"] div.nboutput {
            border: 1px solid #333;
        }
        
        html[data-theme="dark"] .nbinput .highlight,
        html[data-theme="dark"] .nboutput .highlight {
            background-color: #1e1e1e !important;
        }
        html[data-theme="dark"] .highlight pre {
            color: #d4d4d4 !important;
        }
        /* Light mode support for notebooks */
        html[data-theme="light"] .nbinput .highlight,
        html[data-theme="light"] .nboutput .highlight {
            background-color: #f5f5f5 !important;
        }
        html[data-theme="light"] .highlight pre {
            color: #24292e !important;
        }
    </style>
"""

# Additional nbsphinx options
nbsphinx_requirejs_path = ""
nbsphinx_requirejs_options = {}


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
