"""MkDocs hook to render \\full_bibliography as a plain list instead of footnotes.

The mkdocs-bibtex plugin renders \\full_bibliography as markdown footnotes,
which generates broken #fnref back-links when citations live on other pages
(e.g., API reference docstrings). This hook replaces \\full_bibliography with
a clean numbered list before mkdocs-bibtex processes it, avoiding the issue.

Hooks run before plugins, so this intercepts the command first.
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    from pybtex.database import parse_file
except ImportError:
    parse_file = None


def _format_entry(key: str, entry) -> str:
    """Format a single bib entry as a readable string."""
    parts = []

    # Authors
    if "author" in entry.persons:
        authors = entry.persons["author"]
        author_strs = []
        for person in authors:
            first = " ".join(person.first_names)
            last = " ".join(person.last_names)
            author_strs.append(f"{first} {last}" if first else last)
        if len(author_strs) > 3:
            parts.append(f"{author_strs[0]} et al.")
        else:
            parts.append(", ".join(author_strs))

    # Title
    title = entry.fields.get("title", "")
    if title:
        parts.append(f'"{title}"')

    # Journal/booktitle/howpublished
    venue = entry.fields.get("journal", "") or entry.fields.get("booktitle", "") or entry.fields.get("howpublished", "")
    if venue:
        parts.append(f"*{venue}*")

    # Year
    year = entry.fields.get("year", "")
    if year:
        parts.append(f"({year})")

    text = ", ".join(parts) + "." if parts else key

    # DOI or URL link
    doi = entry.fields.get("doi", "")
    url = entry.fields.get("url", "")
    if doi:
        text += f" [doi:{doi}](https://doi.org/{doi})"
    elif url:
        text += f" [link]({url})"

    return text


def on_page_markdown(markdown: str, /, page, config, files, **kwargs) -> str:
    """Replace \\full_bibliography with a plain rendered list."""
    if "\\full_bibliography" not in markdown:
        return markdown

    if parse_file is None:
        return markdown

    # Find bib files
    docs_dir = Path(config["docs_dir"])
    bib_dir = docs_dir  # bibtex plugin uses bib_dir: content, which is the docs_dir
    bib_files = list(bib_dir.glob("*.bib"))

    if not bib_files:
        return markdown

    # Parse all bib entries
    all_entries = {}
    for bf in bib_files:
        try:
            bib_data = parse_file(str(bf))
            all_entries.update(bib_data.entries)
        except Exception:
            continue

    if not all_entries:
        return markdown

    # Build a plain numbered list
    lines = []
    for i, (key, entry) in enumerate(sorted(all_entries.items()), 1):
        formatted = _format_entry(key, entry)
        lines.append(f"{i}. {formatted}")

    bibliography = "\n".join(lines)

    return markdown.replace("\\full_bibliography", bibliography)
