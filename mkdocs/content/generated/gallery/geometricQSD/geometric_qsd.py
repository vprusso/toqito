"""# Geometric quantum state discrimination (qubit SIC-POVM)"""
# %%
# This example demonstrates quantum state discrimination on a highly symmetric
# ensemble: the qubit SIC-POVM (tetrahedral states).
#
# We compute:
#
# - the optimal minimum-error success probability via SDP using
#   :func:`toqito.state_opt.state_distinguishability`,
# - the Pretty Good Measurement (PGM) heuristic via
#   :func:`toqito.measurements.pretty_good_measurement`,
# - geometric diagnostics (pairwise overlaps) and a Bloch-sphere visualization.

import matplotlib.pyplot as plt
import numpy as np

from toqito.measurements import pretty_good_measurement
from toqito.state_opt import state_distinguishability


# %%
# ## Helper functions


def bloch_to_density(n: np.ndarray) -> np.ndarray:
    """Convert a Bloch vector ``n`` in R^3 (with ||n|| <= 1) to a 2x2 density matrix."""
    n = np.asarray(n, dtype=float).reshape(3)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    ident = np.eye(2, dtype=complex)
    return 0.5 * (ident + n[0] * sx + n[1] * sy + n[2] * sz)


def fidelity_pure_states(rho_i: np.ndarray, rho_j: np.ndarray) -> float:
    """For pure states rho=|psi><psi|, return Tr(rho_i rho_j)=|<psi_i|psi_j>|^2."""
    return float(np.real(np.trace(rho_i @ rho_j)))


def calculate_success_prob(states: list[np.ndarray], probs: list[float], povm_ops: list[np.ndarray]) -> float:
    """Compute sum_i p_i Tr(rho_i M_i) for density matrices rho_i and POVM elements M_i."""
    success = 0.0
    for rho, p, meas in zip(states, probs, povm_ops):
        success += p * np.real(np.trace(rho @ meas))
    return float(success)


def plot_bloch_vectors(vectors: np.ndarray, title: str = "Bloch sphere", elev: float = 18, azim: float = 35) -> None:
    """Plot Bloch vectors on top of a Bloch-sphere wireframe."""
    vectors = np.asarray(vectors, dtype=float)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=2, cstride=2, linewidth=0.6, alpha=0.35)

    ax.plot([0, 1], [0, 0], [0, 0], linewidth=1.2)
    ax.plot([0, 0], [0, 1], [0, 0], linewidth=1.2)
    ax.plot([0, 0], [0, 0], [0, 1], linewidth=1.2)

    ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], s=80)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim([-1.05, 1.05])
    ax.set_ylim([-1.05, 1.05])
    ax.set_zlim([-1.05, 1.05])
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


# %%
# ## Qubit SIC-POVM ensemble (tetrahedral states)
#
# A qubit SIC-POVM corresponds to the four vertices of a regular tetrahedron
# on the Bloch sphere.

tetra = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / np.sqrt(3)
states = [bloch_to_density(n_vec) for n_vec in tetra]
probs = [1 / 4] * 4

print("Purities Tr(rho^2):")
for i, rho in enumerate(states):
    print(f"  state {i}: {np.real(np.trace(rho @ rho)):.6f}")

print("\nPairwise overlaps Tr(rho_i rho_j): (1/3 off-diagonal for SIC)")
overlaps = np.zeros((4, 4), dtype=float)
for i in range(4):
    for j in range(4):
        overlaps[i, j] = fidelity_pure_states(states[i], states[j])
print(np.round(overlaps, 6))

plot_bloch_vectors(tetra, title="Qubit SIC (tetrahedral) states on the Bloch sphere")


# %%
# ## Discrimination: optimal vs. PGM

p_best, _ = state_distinguishability(states, probs)
pgm_ops = pretty_good_measurement(states, probs)
p_pgm = calculate_success_prob(states, probs, pgm_ops)

print("\nQubit SIC ensemble (k=4, uniform prior):")
print(f"  P_Best (SDP optimum) = {float(p_best):.6f}")
print(f"  P_PGM  (heuristic)   = {float(p_pgm):.6f}")


# %%
# ## Geometry: overlap matrix visualization

fig = plt.figure(figsize=(5.2, 4.2))
ax = fig.add_subplot(111)
im = ax.imshow(overlaps, interpolation="nearest")
ax.set_title("Overlap matrix: Tr(rho_i rho_j)")
ax.set_xlabel("j")
ax.set_ylabel("i")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
