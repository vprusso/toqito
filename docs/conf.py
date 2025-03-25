# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.insert(0, os.path.abspath('..'))
# sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = "toqito"
copyright = "2020-2025, toqito contributors"
author = "Contributors to toqito"

# The full version, including alpha/beta/rc tags
release = "1.1.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
]

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"
suppress_warnings = ["bibtex.duplicate_label", "bibtex.duplicate_citation"]
# Links matching with the following regular expressions will be ignored
linkcheck_ignore = [
    r"https://arxiv\.org/.*",
    r"https://doi\.org/.*",
    r"https://link\.aps\.org/doi/.*",
    r"http://dx\.doi\.org/.*",
]
# we need to skip these warnigns because all the references appear twice, in a function docstring
# and on the references page.
master_doc = "index"

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
# autosummary_generate = True
# autodoc_typehints = "none"
autoapi_dirs = ["../toqito"]
autoapi_type = "python"
autoapi_ignore = [
    "*/channel_metrics/tests/*",
    "*/rand/tests/*",
    "*/perms/tests/*",
    "*/state_props/tests/*",
    "*/nonlocal_games/tests/*",
    "*/state_metrics/tests/*",
    "*/channel_ops/tests/*",
    "*/helper/tests/*",
    "*/matrix_props/tests/*",
    "*/state_ops/tests/*",
    "*/state_opt/tests/*",
    "*/channels/tests/*",
    "*/matrices/tests/*",
    "*/matrix_ops/tests/*",
    "*/states/tests/*",
    "*/channel_props/tests/*",
    "*/measurements/tests/*",
    "*/measurement_ops/tests/*",
    "*/measurement_props/tests/*",
]
autodoc_typehints = "description"
autoapi_add_toctree_entry = False
autoapi_keep_files = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "furo"
html_logo = "figures/logo.svg"
html_favicon = "figures/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
# html_css_files = ["custom.css"]

# Show in footer when the docs were last updated.
html_last_updated_fmt = "%b %d, %Y"
