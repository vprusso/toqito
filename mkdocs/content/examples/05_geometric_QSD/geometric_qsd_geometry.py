"""# Geometric QSD: trine and tetrahedral state geometry"""
# %%
# ## Geometric QSD
#
# Geometry-first visualizations of two symmetric qubit ensembles:
#
# - trine states (equatorial triangle),
# - SIC tetrahedron (regular tetrahedron).
#
# The side-by-side Bloch plots highlight how state arrangement influences
# distinguishability.

import matplotlib.pyplot as plt
import numpy as np


# %%
# ## Helper functions


def plot_bloch_vectors(
    vectors: np.ndarray,
    title: str = "Bloch vectors",
    ax: plt.Axes | None = None,
    color: str = "tab:blue",
) -> plt.Axes:
    """Draw Bloch sphere and vectors."""
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.12, color="lightgray", linewidth=0)

    vectors = np.asarray(vectors, dtype=float)
    for vec in vectors:
        ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], color=color, lw=2)
    ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], color=color, s=60)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title)
    return ax


# %%
# Trine states on the equator.
angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
trine = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)

# Qubit SIC tetrahedral vectors.
tetra = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / np.sqrt(3)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection="3d")
plot_bloch_vectors(trine, title="Trine ensemble (equatorial triangle)", ax=ax1, color="tab:green")
ax1.view_init(elev=18, azim=35)

ax2 = fig.add_subplot(122, projection="3d")
plot_bloch_vectors(tetra, title="Qubit SIC ensemble (tetrahedron)", ax=ax2, color="tab:purple")
ax2.view_init(elev=18, azim=35)

plt.tight_layout()
plt.show()
