# ruff: noqa: D205, D400, D415
"""CHSH Game Example
=================

.. note::
   This is a sample tutorial to test the layout of the examples gallery. Tutorials are coming soon.

This example calculates the classical and quantum values for the CHSH game
using the toqito library. It demonstrates how to construct the necessary matrices
and use the XORGame function.
"""

# sphinx_gallery_thumbnail_path = '_static/thumbnail.png'

import numpy as np

r = np.array([[1, 0], [0, 1]])
r = np.kron(r, r)
print(r)
