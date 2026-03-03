"""# Geometric QSD performance comparison: optimal SDP vs PGM"""
# %%
# ## Performance comparison
#
# We scan a one-parameter family of binary ensembles and compare:
#
# - optimal SDP value (`state_distinguishability`),
# - PGM success probability (`pretty_good_measurement`).
#
# Family used: two pure qubit states with equal priors, where the Bloch angle
# ``theta in [0, pi]`` controls distinguishability.

import matplotlib.pyplot as plt
import numpy as np

from toqito.measurements import pretty_good_measurement
from toqito.state_opt import state_distinguishability


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


def pure_state_from_bloch(theta: float) -> np.ndarray:
    """Return a pure qubit density matrix with Bloch vector (sin(theta), 0, cos(theta))."""
    n_vec = np.array([np.sin(theta), 0.0, np.cos(theta)])
    return bloch_to_density(n_vec)


# %%
thetas = np.linspace(0, np.pi, 31)
opt_vals, pgm_vals = [], []

for theta in thetas:
    rho_0 = pure_state_from_bloch(0.0)
    rho_1 = pure_state_from_bloch(theta)
    states = [rho_0, rho_1]
    probs = [0.5, 0.5]

    p_opt, _ = state_distinguishability(states, probs)
    m_pgm = pretty_good_measurement(states, probs)
    p_pgm = calculate_success_prob(states, probs, m_pgm)

    opt_vals.append(float(p_opt))
    pgm_vals.append(float(p_pgm))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(thetas, opt_vals, lw=2, label="Optimal SDP (state_distinguishability)")
ax.plot(thetas, pgm_vals, "--", lw=2, label="PGM (pretty_good_measurement)")
ax.set_xlabel(r"Bloch angle $\\theta$ between states")
ax.set_ylabel("Success probability")
ax.set_title("Performance comparison")
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()
