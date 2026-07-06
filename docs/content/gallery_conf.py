"""mkdocs-gallery configuration for the example gallery.

This sets the order in which the gallery subsections are listed on the gallery index:

    Basic Tutorials -> Quantum States -> Nonlocal Games -> Extended Nonlocal Games

mkdocs-gallery 0.10.4 (the latest release) has a bug in ``Gallery.populate_subsections``:
the subsection sort key is wrapped in a nested function that calls itself instead of the
underlying key, so any non-``None`` ``subsection_order`` raises ``RecursionError`` (and the
key is handed a ``str`` rather than the ``Path`` that ``ExplicitOrder`` expects). Until that
is fixed upstream, replace the method with a corrected copy that applies the sort key
directly. Once mkdocs-gallery ships a fix, this monkeypatch can be dropped and only the
``conf`` dict below is needed.

The same release also parses module docstrings with ``ast.Str`` and the ``.s`` attribute in
``py_source_parser._get_docstring_and_rest``. Both were deprecated in Python 3.12 (emitting the
``ast.Str`` / ``.s`` ``DeprecationWarning`` seen in strict docs builds) and are removed in
Python 3.14, where the gallery build would raise ``AttributeError`` instead. Those three names
are the only ``ast.Str`` / ``.s`` uses anywhere in mkdocs-gallery, so a corrected copy of that
one function using ``ast.Constant`` / ``.value`` both silences the warning today and unblocks
Python 3.14. This monkeypatch can likewise be dropped once mkdocs-gallery ships a fix.
"""

import ast
import tokenize
from io import BytesIO

from mkdocs_gallery import py_source_parser
from mkdocs_gallery.errors import ExtensionError
from mkdocs_gallery.gen_data_model import Gallery, GallerySubSection, _has_readme
from mkdocs_gallery.py_source_parser import SYNTAX_ERROR_DOCSTRING, parse_source_file
from mkdocs_gallery.sorting import ExplicitOrder

# Pedagogical order rather than the alphabetical default. Every subsection directory must be
# listed here (ExplicitOrder raises for any folder it does not find).
SUBSECTION_ORDER = ExplicitOrder(
    [
        "basics",
        "quantum_states",
        "nonlocal_games",
        "extended_nonlocal_games",
    ]
)


def _populate_subsections(self) -> None:
    """Corrected copy of ``Gallery.populate_subsections`` (mkdocs-gallery 0.10.4 bugfix)."""
    assert self.subsections is None, "This method can only be called once !"  # noqa: S101

    subfolders = [
        subfolder for subfolder in self.scripts_dir.iterdir() if subfolder.is_dir() and _has_readme(subfolder)
    ]

    sortkey = self.conf["subsection_order"]
    sorted_subfolders = sorted(subfolders, key=sortkey) if sortkey is not None else sorted(subfolders)

    self.subsections = tuple(
        GallerySubSection(self, subpath=f.relative_to(self.scripts_dir)) for f in sorted_subfolders
    )


Gallery.populate_subsections = _populate_subsections


def _get_docstring_and_rest(file):
    """Corrected copy of ``py_source_parser._get_docstring_and_rest`` (mkdocs-gallery 0.10.4 bugfix).

    Identical to the upstream function on Python 3.12/3.13 but uses ``ast.Constant`` / ``.value``
    in place of the deprecated ``ast.Str`` / ``.s``, so it is silent on 3.12/3.13 and does not
    ``AttributeError`` on Python 3.14, where ``ast.Str`` has been removed. The upstream
    Python < 3.7 fallback branch is dropped since toqito requires Python >= 3.12.
    """
    node, content = parse_source_file(file)

    if node is None:
        return SYNTAX_ERROR_DOCSTRING, content, 1, node

    if not isinstance(node, ast.Module):
        raise ExtensionError(f"This function only supports modules. You provided {node.__class__.__name__}")
    if not (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        raise ExtensionError(
            f'Could not find docstring in file "{file}". '
            "A docstring is required by mkdocs-gallery "
            'unless the file is ignored by "ignore_pattern"'
        )

    docstring = ast.get_docstring(node)
    assert docstring is not None, f'Could not find docstring in file "{file}".'  # noqa: S101
    # Preserve mkdocs-gallery's backward-compatible handling of a leading newline.
    if len(node.body[0].value.value) and node.body[0].value.value[0] == "\n":
        docstring = "\n" + docstring
    ts = tokenize.tokenize(BytesIO(content.encode()).readline)
    # Find the first string token and take its end row as the docstring's last line.
    for tk in ts:
        if tk.exact_type == tokenize.STRING:
            lineno, _ = tk.end
            break
    else:
        lineno = 0

    rest = "\n".join(content.split("\n")[lineno:])
    lineno += 1
    return docstring, rest, lineno, node


py_source_parser._get_docstring_and_rest = _get_docstring_and_rest

# remove_config_comments strips gallery directives (e.g. mkdocs_gallery_thumbnail_path)
# from the rendered pages instead of showing them as literal text.
conf = {"subsection_order": SUBSECTION_ORDER, "remove_config_comments": True}
