"""MkDocs hook to order the Examples gallery subsections in the navigation.

The ``- Examples: generated/gallery`` nav entry is auto-populated by mkdocs from the generated
gallery directory, which lists the subsections alphabetically (Basic Tutorials, Extended
Nonlocal Games, Nonlocal Games, Quantum States). This reorders the "Examples" section's
children into a pedagogical order.

mkdocs-gallery's ``subsection_order`` (see ``content/gallery_conf.py``) only affects the gallery
index page, not this sidebar navigation, so the ordering has to be applied here as well.
"""

from mkdocs.structure.nav import Navigation, Section

# Desired order of the Examples subsections, by their rendered titles.
_EXAMPLES_ORDER = (
    "Basic Tutorials",
    "Quantum States",
    "Nonlocal Games",
    "Extended Nonlocal Games",
)


def on_nav(nav: Navigation, /, **kwargs) -> Navigation:
    """Reorder the children of the top-level 'Examples' section."""
    rank = {title: index for index, title in enumerate(_EXAMPLES_ORDER)}
    for item in nav.items:
        if isinstance(item, Section) and item.title == "Examples":
            # Unknown children (should be none) sort after the known subsections, preserving
            # their relative order via Python's stable sort.
            item.children.sort(key=lambda child: rank.get(getattr(child, "title", None), len(rank)))
            break
    return nav
