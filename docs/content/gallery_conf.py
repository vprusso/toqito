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
"""

from mkdocs_gallery.gen_data_model import Gallery, GallerySubSection, _has_readme
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

conf = {"subsection_order": SUBSECTION_ORDER}
