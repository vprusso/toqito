"""# SIC-POVM intuition and discrimination with toqito"""
# %%
# ## Preamble
#
# This tutorial builds geometric intuition for minimum-error quantum state
# discrimination (QSD) and demonstrates practical workflows in `toqito`:
#
# - `toqito.state_opt.state_distinguishability` for SDP-based optimal success
#   probability,
# - `toqito.measurements.pretty_good_measurement` for the PGM heuristic,
# - Bloch-sphere and overlap visualizations for symmetric ensembles.

import matplotlib.pyplot as plt
import numpy as np

from toqito.measurements import pretty_good_measurement
from toqito.state_opt import state_distinguishability


# %%
# ## SIC-POVM background and geometry
#
# A SIC-POVM (Symmetric Informationally Complete POVM) in dimension ``d`` is a
# set of ``d^2`` rank-1 effects
#
# ``E_i = (1 / d) |psi_i><psi_i|`` for ``i = 1, ..., d^2``
#
# with constant pairwise overlaps
#
# ``|<psi_i|psi_j>|^2 = 1 / (d + 1)`` for ``i != j``.
#
# Why this matters for QSD examples:
#
# - SIC ensembles are maximally symmetric and geometrically clean.
# - The overlap structure is explicit and easy to visualize.
# - They are a useful benchmark for comparing optimal SDP discrimination
#   against PGM.
#
# For qubits (``d = 2``), a SIC corresponds to four pure states whose Bloch
# vectors are the vertices of a regular tetrahedron.


# %%
# ## Helper functions


def bloch_to_density(n: np.ndarray) -> np.ndarray:
    """Map a Bloch vector ``n`` in R^3 to a qubit density matrix."""
    n = np.asarray(n, dtype=float).reshape(3)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    return 0.5 * (ident + n[0] * sx + n[1] * sy + n[2] * sz)


def calculate_success_prob(states: list[np.ndarray], probs: list[float], povm_ops: list[np.ndarray]) -> float:
    """Compute ``sum_i p_i Tr[M_i rho_i]``."""
    return float(np.real(sum(p * np.trace(meas @ rho) for p, meas, rho in zip(probs, povm_ops, states))))


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
# ## Qubit SIC example (tetrahedron)
#
# We choose tetrahedral Bloch vectors
#
# ``(1,1,1)/sqrt(3), (1,-1,-1)/sqrt(3), (-1,1,-1)/sqrt(3), (-1,-1,1)/sqrt(3)``
#
# which satisfy the qubit SIC condition ``Tr(rho_i rho_j) = 1/3`` for all
# ``i != j``.

tetra = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / np.sqrt(3)
states_sic = [bloch_to_density(n_vec) for n_vec in tetra]
probs_sic = [1 / 4] * 4

overlaps_sic = np.array([[np.real(np.trace(a @ b)) for b in states_sic] for a in states_sic])

print("Purities Tr(rho_i^2):")
for i, rho in enumerate(states_sic):
    print(f"  state {i}: {np.real(np.trace(rho @ rho)):.6f}")

print("\nPairwise overlaps Tr(rho_i rho_j):")
print(np.round(overlaps_sic, 6))

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
plot_bloch_vectors(tetra, title="Qubit SIC states (tetrahedral vertices)", ax=ax, color="tab:purple")
plt.tight_layout()
plt.show()


# %%
# ## Discrimination with toqito: optimal vs PGM

p_best_sic, _ = state_distinguishability(states_sic, probs_sic)
pgm_sic = pretty_good_measurement(states_sic, probs_sic)
p_pgm_sic = calculate_success_prob(states_sic, probs_sic, pgm_sic)

print(f"SIC ensemble: P_best = {float(p_best_sic):.6f}")
print(f"SIC ensemble: P_pgm  = {float(p_pgm_sic):.6f}")
