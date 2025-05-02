Examples
========

.. warning::
   This page is currently a Work in Progress. Example tutorials will be added in the future.

Contributing to example tutorials
=================================

   For full details, see the `Sphinx‑Gallery Quickstart`_.

   .. _Sphinx‑Gallery Quickstart: https://sphinx-gallery.github.io/stable/index.html

   Here’s a checklist for adding a new example script:

   1. Place your script under the `examples/` directory.
   2. Add a module‐level docstring for title and description:

      .. code-block:: python

         """
         My Example Title
         =================

         A short description of what this example demonstrates.
         """

   3. Use cell delimiters (e.g. `# %%`) to break your script into logical sections:

      .. code-block:: python

         # %% [markdown]
         # Section heading

         # %% [python]
         import numpy as np
         # ... your code ...

   4. Specify which figure to use as the thumbnail:

      - By explicit path:

        .. code-block:: python

           # sphinx_gallery_thumbnail_path = '_static/my_thumb.png'

      - If no thumbnail is produced as a plot or linked, the default thumbnail (toqito logo) will be displayed.

   5. If you want to implement subcategories, create a new directory under `examples/`, include a `README.rst` file, and add all the Python script examples that fall into that subcategory.

   This formatting ensures your example runs smoothly and renders correctly in the gallery.

   We welcome contributions from the community!
