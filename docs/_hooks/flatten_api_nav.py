"""MkDocs hook to remove the outer 'toqito' nesting layer from API Reference.

The api-autonav plugin generates: API Reference > toqito > toqito.channels, ...
This hook flattens it to:          API Reference > toqito.channels, ...
"""

from mkdocs.structure.nav import Navigation, Section


def on_nav(nav: Navigation, /, **kwargs) -> Navigation:
    for item in nav.items:
        if isinstance(item, Section) and item.title == "API Reference":
            # Find the single "toqito" child section and promote its children
            if (
                len(item.children) == 1
                and isinstance(item.children[0], Section)
            ):
                wrapper = item.children[0]
                for child in wrapper.children:
                    child.parent = item
                item.children = wrapper.children
            break
    return nav
