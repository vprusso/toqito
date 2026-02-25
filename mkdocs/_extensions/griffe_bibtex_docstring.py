"""Griffe extension to resolve [@cite_key] references in docstrings using bibtex.

This extension pre-processes docstrings before mkdocstrings renders them,
replacing [@cite_key] with clickable superscript numbered references that
link to a numbered reference list at the bottom of the docstring. This is
necessary because mkdocs-bibtex only processes page-level markdown
(on_page_markdown), but mkdocstrings renders docstrings in a separate Markdown
pipeline that bibtex never sees.
"""

from __future__ import annotations

import re
from pathlib import Path

import griffe


def _load_bib_entries(bib_dir: str | Path) -> dict[str, str]:
    """Load all .bib files from a directory and return a dict of key -> formatted reference."""
    try:
        from pybtex.database import parse_file
    except ImportError:
        # Fallback: try pypandoc-based parsing
        entries = {}
        bib_path = Path(bib_dir)
        if bib_path.is_file():
            bib_files = [bib_path]
        elif bib_path.is_dir():
            bib_files = list(bib_path.glob("*.bib"))
        else:
            return entries

        for bf in bib_files:
            current_key = None
            for line in bf.read_text(encoding="utf-8").splitlines():
                m = re.match(r"@\w+\{(.+),\s*$", line)
                if m:
                    current_key = m.group(1).strip()
                    entries[current_key] = current_key  # placeholder
        return entries

    entries = {}
    bib_path = Path(bib_dir)
    if bib_path.is_file():
        bib_files = [bib_path]
    elif bib_path.is_dir():
        bib_files = list(bib_path.glob("*.bib"))
    else:
        return entries

    for bf in bib_files:
        try:
            bib_data = parse_file(str(bf))
        except Exception:
            continue
        for key, entry in bib_data.entries.items():
            # Build a simple formatted reference string
            parts = []

            # Authors
            if "author" in entry.persons:
                authors = entry.persons["author"]
                author_strs = []
                for person in authors:
                    first = " ".join(person.first_names)
                    last = " ".join(person.last_names)
                    author_strs.append(f"{last}, {first}" if first else last)
                parts.append(" and ".join(author_strs))

            # Title
            title = entry.fields.get("title", "")
            if title:
                parts.append(f"*{title}*")

            # Journal/booktitle
            journal = entry.fields.get("journal", entry.fields.get("booktitle", ""))
            if journal:
                parts.append(f"**{journal}**")

            # Volume
            volume = entry.fields.get("volume", "")
            if volume:
                number = entry.fields.get("number", "")
                vol_str = f"vol. {volume}"
                if number:
                    vol_str += f"({number})"
                parts.append(vol_str)

            # Year
            year = entry.fields.get("year", "")
            if year:
                parts.append(f"({year})")

            # DOI/URL
            doi = entry.fields.get("doi", "")
            url = entry.fields.get("url", "")
            if doi:
                parts.append(f"[doi:{doi}](https://doi.org/{doi})")
            elif url:
                parts.append(f"[link]({url})")

            entries[key] = ". ".join(parts) + "."

    return entries


# Module-level cache for bib entries
_bib_cache: dict[str, str] | None = None
_bib_dir: str | None = None


class BibtexDocstringExtension(griffe.Extension):
    """Griffe extension that resolves [@cite_key] in docstrings."""

    def __init__(self, bib_dir: str = "content") -> None:
        """Initialize with the path to the bib directory.

        Args:
            bib_dir: Path to directory containing .bib files,
                     relative to the mkdocs config directory.
        """
        global _bib_cache, _bib_dir

        # Resolve bib_dir relative to this extension file's location
        # which is in mkdocs/_extensions/
        mkdocs_dir = Path(__file__).parent.parent
        resolved = mkdocs_dir / bib_dir
        if not resolved.exists():
            # Try from current working directory
            resolved = Path(bib_dir)

        _bib_dir = str(resolved)
        _bib_cache = None  # Reset cache

    def _get_entries(self) -> dict[str, str]:
        global _bib_cache
        if _bib_cache is None:
            _bib_cache = _load_bib_entries(_bib_dir)
        return _bib_cache

    def _process_docstring(self, obj: griffe.Object) -> None:
        """Process a Griffe object's docstring to resolve bibtex citations."""
        if obj.docstring is None:
            return

        text = obj.docstring.value
        if "[@" not in text:
            return

        entries = self._get_entries()

        # Build a unique prefix for anchor IDs to avoid collisions on the same page
        obj_path = obj.path.replace(".", "-").replace("/", "-")

        # Find all [@key] patterns
        cite_pattern = re.compile(r"\[@([\w]+(?:;@[\w]+)*)\]")
        found_keys: list[str] = []
        # Map from cite key to its assigned number (per-docstring numbering)
        key_to_num: dict[str, int] = {}
        counter = 0

        def replace_cite(match: re.Match) -> str:
            nonlocal counter
            keys_str = match.group(1)
            keys = [k.lstrip("@") for k in keys_str.split(";")]
            refs = []
            for key in keys:
                if key not in key_to_num:
                    counter += 1
                    key_to_num[key] = counter
                found_keys.append(key)
                num = key_to_num[key]
                # Clickable superscript that jumps to the reference at the bottom
                refs.append(
                    f'<sup><a id="cite-{obj_path}-{num}" href="#ref-{obj_path}-{num}">{num}</a></sup>'
                )
            return "".join(refs)

        new_text = cite_pattern.sub(replace_cite, text)

        # Append numbered reference list with back-links
        references = []
        seen: set[str] = set()
        for key in found_keys:
            if key in seen:
                continue
            seen.add(key)
            num = key_to_num[key]
            ref_text = entries[key] if key in entries else key
            references.append(
                f'<span id="ref-{obj_path}-{num}">'
                f'<sup><a href="#cite-{obj_path}-{num}">{num}</a></sup> {ref_text}'
                f"</span>"
            )

        if references:
            new_text = new_text + "\n\n**References**\n\n" + "<br>\n".join(references)

        obj.docstring.value = new_text

    def on_function_instance(self, *, node, func: griffe.Function, **kwargs) -> None:
        """Process function docstrings."""
        self._process_docstring(func)

    def on_class_instance(self, *, node, cls: griffe.Class, **kwargs) -> None:
        """Process class docstrings."""
        self._process_docstring(cls)

    def on_module_instance(self, *, node, mod: griffe.Module, **kwargs) -> None:
        """Process module docstrings."""
        self._process_docstring(mod)

    def on_attribute_instance(self, *, node, attr: griffe.Attribute, **kwargs) -> None:
        """Process attribute docstrings."""
        self._process_docstring(attr)
