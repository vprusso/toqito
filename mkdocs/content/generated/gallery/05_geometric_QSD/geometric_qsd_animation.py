"""# Geometric QSD animation: Bloch-angle sweep for binary discrimination"""
# %%
# ## Animation section
#
# This section adds an animation-style view of the same one-parameter binary
# discrimination family used in the performance comparison example.
#
# At each frame, we:
#
# - update a second qubit state on the Bloch sphere,
# - solve for the optimal SDP success probability,
# - evaluate the PGM success probability,
# - display both values while the Bloch angle ``theta`` changes.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

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


def pure_state_from_bloch(theta: float) -> np.ndarray:
    """Return a pure qubit density matrix with Bloch vector (sin(theta), 0, cos(theta))."""
    n_vec = np.array([np.sin(theta), 0.0, np.cos(theta)])
    return bloch_to_density(n_vec)


def calculate_success_prob(states: list[np.ndarray], probs: list[float], povm_ops: list[np.ndarray]) -> float:
    """Compute ``sum_i p_i Tr[M_i rho_i]``."""
    return float(np.real(sum(p * np.trace(meas @ rho) for p, meas, rho in zip(probs, povm_ops, states))))


# %%
# ## Build animation data

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


# %%
# ## Animate Bloch-angle sweep and running performance curves

fig = plt.figure(figsize=(12, 5))
ax_bloch = fig.add_subplot(121, projection="3d")
ax_perf = fig.add_subplot(122)

# Bloch sphere wireframe
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 20)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))
ax_bloch.plot_wireframe(x, y, z, rstride=2, cstride=2, linewidth=0.6, alpha=0.3)
ax_bloch.set_xlim([-1.05, 1.05])
ax_bloch.set_ylim([-1.05, 1.05])
ax_bloch.set_zlim([-1.05, 1.05])
ax_bloch.set_title("Binary-state geometry")
ax_bloch.set_xlabel("x")
ax_bloch.set_ylabel("y")
ax_bloch.set_zlabel("z")

# Fixed state rho0 at north pole, moving rho1 with theta.
line0, = ax_bloch.plot([0, 0], [0, 0], [0, 1], lw=2, color="tab:blue")
pt0 = ax_bloch.scatter([0], [0], [1], color="tab:blue", s=60, label=r"$\\rho_0$")
line1, = ax_bloch.plot([], [], [], lw=2, color="tab:orange")
pt1 = ax_bloch.scatter([], [], [], color="tab:orange", s=60, label=r"$\\rho_1(\\theta)$")
ax_bloch.legend(loc="upper left")

# Performance panel
ax_perf.set_xlim(0, np.pi)
ax_perf.set_ylim(0.45, 1.02)
ax_perf.set_xlabel(r"Bloch angle $\\theta$")
ax_perf.set_ylabel("Success probability")
ax_perf.set_title("Running SDP vs PGM")
ax_perf.grid(alpha=0.25)
line_opt, = ax_perf.plot([], [], lw=2, label="Optimal SDP")
line_pgm, = ax_perf.plot([], [], "--", lw=2, label="PGM")
ax_perf.legend(loc="lower right")
text = ax_perf.text(0.02, 0.05, "", transform=ax_perf.transAxes)


def update(frame: int):
    """Update function for animation frame index."""
    theta = thetas[frame]
    vec1 = np.array([np.sin(theta), 0.0, np.cos(theta)])

    line1.set_data([0, vec1[0]], [0, vec1[1]])
    line1.set_3d_properties([0, vec1[2]])

    # Re-draw moving point by replacing the artist.
    global pt1
    pt1.remove()
    pt1 = ax_bloch.scatter([vec1[0]], [vec1[1]], [vec1[2]], color="tab:orange", s=60)

    line_opt.set_data(thetas[: frame + 1], opt_vals[: frame + 1])
    line_pgm.set_data(thetas[: frame + 1], pgm_vals[: frame + 1])

    text.set_text(
        rf"$\\theta={theta:.2f}$ rad\n"
        rf"$P_\\mathrm{{opt}}={opt_vals[frame]:.4f}$\n"
        rf"$P_\\mathrm{{PGM}}={pgm_vals[frame]:.4f}$"
    )
    return line1, line_opt, line_pgm, text, pt1


anim = FuncAnimation(fig, update, frames=len(thetas), interval=250, blit=False, repeat=True)
plt.tight_layout()
plt.show()
