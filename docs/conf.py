# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import datetime
import os
import re
import sys
from pathlib import Path
from sphinx_gallery.sorting import ExplicitOrder

# sys.path.insert(0, os.path.abspath("."))
# sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))

DOCS_DIR = Path(__file__).resolve().parent

# -- Project information -----------------------------------------------------

project = "|toqito>"
copyright = f"2020 - {datetime.date.today().year}, |toqito> contributors"
author = "|toqito> contributors"

# The full version, including alpha/beta/rc tags
release = "1.1.2"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "jupyter_sphinx",
    "sphinx.ext.duration",
]


DOCS_FAST = bool(os.environ.get("TOQITO_DOCS_FAST"))


sphinx_gallery_conf = {
    "examples_dirs": "examples",  # Path to example scripts
    "gallery_dirs": "auto_examples",  # Output directory for generated example galleries
    "subsection_order": ExplicitOrder(
        [
            "examples/basics",
            "examples/quantum_states",
            "examples/nonlocal_games",
            "examples/extended_nonlocal_games",
        ]
    ),
    # Match every example script; filenames no longer share a common prefix.
    "filename_pattern": r"^",
    "write_computation_times": False,  # Do not include computation times
    "default_thumb_file": str(DOCS_DIR / "figures" / "logo.png"),  # Default thumbnail image
    "line_numbers": True,  # add line numbers
    "download_all_examples": False,
    "ignore_pattern": r"__init__\.py",
}

autoapi_options = [
    "undoc-members",
    "show-inheritance",
    "imported-members",
]


bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"
# Links matching with the following regular expressions will be ignored
linkcheck_ignore = [
    r"https://arxiv\.org/.*",
    r"https://doi\.org/.*",
    r"https://link\.aps\.org/doi/.*",
    r"http://dx\.doi\.org/.*",
    r"https://www\.quantiki\.org/.*",
]
# we need to skip these warnigns because all the references appear twice, in a function docstring
# and on the references page.
master_doc = "index"

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
autoapi_add_toctree_entry = True
autoapi_keep_files = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "_templates",
    "Thumbs.db",
    ".DS_Store",
]


if DOCS_FAST:
    sphinx_gallery_conf["run_stale_examples"] = False


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
# Add jupyter configuration

# Show in footer when the docs were last updated.
html_last_updated_fmt = "%b %d, %Y"


def setup(app):
    # light docs build option
    app.connect("autodoc-process-docstring", maybe_strip_jupyter_blocks)

    # Add ':orphan:' to example docs to avoid toctree warnings
    app.connect("source-read", add_orphan_to_examples)

    # Create CSS that inherits colors from Furo theme
    app.add_css_file("jupyter-sphinx-override.css")
    css_content = """
    /* Override jupyter-sphinx styling to match Furo theme */
    div.jupyter_container {
        background-color: var(--color-background-secondary);
        border: 1px solid var(--color-background-border);
        box-shadow: none;
        margin-bottom: 1em;
    }
    
    .jupyter_container div.code_cell {
        background-color: var(--color-code-background);
        border: 1px solid var(--color-background-border);
        border-radius: 0.2rem;
    }
    
    .jupyter_container div.code_cell pre {
        background-color: var(--color-code-background);
        color: var(--color-code-foreground);
        font-family: var(--font-stack--monospace);
    }
    
    div.jupyter_container div.highlight {
        background-color: var(--color-code-background);
    }
    
    .jupyter_container .output {
        background-color: var(--color-background-secondary);
        padding: 0.5em;
        border-top: 1px solid var(--color-background-border);
    }
    
    .jupyter_container div.output pre {
        background-color: var(--color-background-secondary);
        color: var(--color-foreground-primary);
        font-family: var(--font-stack--monospace);
    }
    
    /* Fix for output highlighting */
    .jupyter_container .output .highlight {
        background-color: var(--color-background-secondary);
    }
    
    /* Style for the prompt */
    .jupyter_container .highlight .gp {
        color: var(--color-brand-primary);
        font-weight: bold;
    }
    
    /* Style for code comments */
    .jupyter_container .highlight .c1 {
        color: var(--color-foreground-secondary);
        font-style: italic;
    }
    """
    static_dir = os.path.join(app.outdir, "_static")
    os.makedirs(static_dir, exist_ok=True)

    with open(os.path.join(static_dir, "jupyter-sphinx-override.css"), "w") as f:
        f.write(css_content)


FAST_MODE = os.environ.get("TOQITO_DOCS_FAST") == "1"
ONLY_DOC_TARGET = os.environ.get("TOQITO_DOCS_ONLY", "")


def maybe_strip_jupyter_blocks(app, what, name, obj, options, lines):
    # If not fast mode or no jupyter blocks, skip
    if not FAST_MODE:
        return

    # if only want to run one docstring example
    should_run = name == ONLY_DOC_TARGET
    in_block = False
    new_lines = []

    for line in lines:
        # modify the code blocks to readable code blocks only during build time
        if ".. jupyter-execute::" in line:
            in_block = True
            if should_run:
                new_lines.append(line)
            else:
                new_lines.append(".. code-block:: python")
            continue
        elif in_block:
            # Skip jupyter-execute options (like :raises-exception:)
            if re.match(r"\s*:[a-zA-Z-]+:", line):
                continue
            if not line.strip() or len(line) - len(line.lstrip()) < 3:
                in_block = False
        new_lines.append(line)

    lines[:] = new_lines


def add_orphan_to_examples(app, docname, source):
    """
    Prepend ':orphan:' to example .rst files generated by sphinx-gallery.
    """
    if docname.startswith("examples/") or "/examples/" in docname:
        if not source[0].lstrip().startswith(":orphan:"):
            source[0] = ":orphan:\n\n" + source[0]
