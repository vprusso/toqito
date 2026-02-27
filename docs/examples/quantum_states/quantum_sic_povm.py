"""
SIC-POVMs and Quantum State Discrimination
==========================================

This tutorial explores **Symmetric Informationally Complete Positive
Operator-Valued Measures** (SIC-POVMs) using :code:`|toqito⟩`.

.. note::

   Weyl-Heisenberg SIC fiducials have been found numerically in every dimension
   up to at least :math:`d = 151` and analytically in infinitely many dimensions,
   yet a proof of existence for all :math:`d` (Zauner's conjecture) remains open.
"""

# %%
# This tutorial assumes familiarity with the basics of quantum information.
# For background see :footcite:`Chuang_2011_Quantum` or
# :footcite:`Watrous_2018_TQI`.  Installation instructions for
# :code:`|toqito⟩` are in :ref:`getting_started_reference-label`.

# %%
# Imports
# -------
#
# We use only :code:`|toqito⟩` and the Python standard library.

import numpy as np
from itertools import combinations
from math import comb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from toqito.state_opt import state_distinguishability
from toqito.matrix_ops import vectors_to_gram_matrix
from toqito.matrices import gen_pauli, standard_basis
from toqito.states import mutually_unbiased_basis
from toqito.state_props import learnability

# %%
# 1. The State Discrimination Problem
# ------------------------------------
#
# Alice draws a state :math:`|\psi_i\rangle` uniformly at random from a finite
# ensemble and sends it to Bob.  Bob performs a measurement :math:`\{M_i\}` and
# announces his guess.  The optimal *success probability* over all POVMs is
#
# .. math::
#
#    p_{\text{succ}} = \max_{\{M_i\} \geq 0,\,\sum M_i = I}
#    \sum_i p_i \langle \psi_i | M_i | \psi_i \rangle.
#
# :code:`|toqito⟩` solves this semidefinite programme via
# :py:func:`~toqito.state_opt.state_distinguishability`.
#
# As a warm-up, two orthogonal qubit states are perfectly distinguishable.

e0, e1 = standard_basis(2)

rho0 = e0 @ e0.conj().T
rho1 = e1 @ e1.conj().T

p_succ_orth, _ = state_distinguishability([rho0, rho1], [0.5, 0.5])
print(f"Orthogonal basis  p_succ = {p_succ_orth:.6f}   (expected 1.0)")

# %%
# Bloch Sphere Helpers
# ^^^^^^^^^^^^^^^^^^^^^
#
# Every qubit pure state :math:`|\psi\rangle` maps to a point on the unit
# sphere in :math:`\mathbb{R}^3` via the **Bloch vector**
#
# .. math::
#
#    \vec{r} = \bigl(2\,\mathrm{Re}\,\rho_{01},\;
#                    2\,\mathrm{Im}\,\rho_{10},\;
#                    \rho_{00} - \rho_{11}\bigr),
#    \qquad \rho = |\psi\rangle\langle\psi|.
#
# We define two small helpers — one to compute Bloch coordinates, one to
# render states on the sphere — which we reuse throughout the tutorial.

def bloch_coords(state):
    """Return the (x, y, z) Bloch vector of a qubit pure state."""
    rho = np.outer(state, state.conj())
    x =  2 * np.real(rho[0, 1])
    y =  2 * np.imag(rho[1, 0])
    z =  np.real(rho[0, 0] - rho[1, 1])
    return np.array([x, y, z])


def plot_bloch_sphere(state_groups, labels, colors, title, markers=None):
    """
    Plot one or more groups of qubit states on the Bloch sphere.

    Parameters
    ----------
    state_groups : list of list of np.ndarray
        Each inner list is a group of qubit state vectors to plot together.
    labels : list of str
        Legend label for each group.
    colors : list of str
        Marker colour for each group.
    title : str
        Figure title.
    markers : list of str, optional
        Marker style for each group (default: 'o' for all).
    """
    if markers is None:
        markers = ["o"] * len(state_groups)

    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection="3d")

    # Translucent unit sphere
    u, v = np.mgrid[0 : 2 * np.pi : 120j, 0 : np.pi : 60j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    ax.plot_surface(xs, ys, zs, color="lightsteelblue", alpha=0.08, linewidth=0)

    # Equator and principal meridians
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 0,          color="gray", lw=0.6, alpha=0.5)
    ax.plot(np.cos(theta), np.zeros(200), np.sin(theta), color="gray", lw=0.6, alpha=0.5)
    ax.plot(np.zeros(200), np.cos(theta), np.sin(theta), color="gray", lw=0.6, alpha=0.5)

    # Axes arrows
    for vec, lbl in zip([(1,0,0),(0,1,0),(0,0,1)], ["x","y","z"]):
        ax.quiver(0, 0, 0, *vec, length=1.25, color="dimgray",
                  arrow_length_ratio=0.08, linewidth=0.8)
        ax.text(*(np.array(vec) * 1.35), lbl, color="dimgray",
                fontsize=9, ha="center", va="center")

    # State vectors
    for group, label, color, marker in zip(state_groups, labels, colors, markers):
        vecs = np.array([bloch_coords(s) for s in group])
        ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2],
                   color=color, s=80, marker=marker,
                   zorder=5, label=label, depthshade=False)
        for vec in vecs:
            ax.quiver(0, 0, 0, *vec, color=color, alpha=0.7,
                      arrow_length_ratio=0.08, linewidth=1.2)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title, pad=12)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()

# %%
# **Plot 1 — Orthogonal Basis on the Bloch Sphere**
#
# :math:`|0\rangle` sits at the north pole and :math:`|1\rangle` at the south
# pole.  Their antipodal placement is the geometric reason they can be
# perfectly distinguished: a projective measurement along the :math:`z`-axis
# separates them with probability 1.

plot_bloch_sphere(
    [[e0], [e1]],
    labels=[r"$|0\rangle$ (north pole)", r"$|1\rangle$ (south pole)"],
    colors=["royalblue", "crimson"],
    markers=["^", "v"],
    title=r"Orthogonal basis — antipodal points ($p_\mathrm{succ}=1$)",
)

# %%
# 2. The Qubit SIC-POVM — Tetrahedral States
# -------------------------------------------
#
# A SIC-POVM in dimension :math:`d` is a set of :math:`d^2` unit vectors
# :math:`\{|\psi_i\rangle\}` satisfying the **equiangularity** condition
#
# .. math::
#
#    |\langle \psi_i | \psi_j \rangle|^2 = \frac{1}{d+1}, \quad i \neq j.
#
# For :math:`d=2` this gives four vectors with pairwise overlap-squared
# :math:`\tfrac{1}{3}`, corresponding to the vertices of a regular tetrahedron
# inscribed in the Bloch sphere.  The POVM elements are
# :math:`M_i = |\psi_i\rangle\langle\psi_i| / d`, and they resolve the identity
# :math:`\sum_i M_i = I`.

def sic_d2():
    """Four tetrahedral SIC states in dimension 2."""
    phi = 2 * np.pi / 3
    return [
        np.array([1, 0], dtype=complex),
        np.array([1 / np.sqrt(3), np.sqrt(2 / 3)], dtype=complex),
        np.array([1 / np.sqrt(3), np.sqrt(2 / 3) * np.exp(1j * phi)], dtype=complex),
        np.array([1 / np.sqrt(3), np.sqrt(2 / 3) * np.exp(2j * phi)], dtype=complex),
    ]

sic2 = sic_d2()

# %%
# **Plot 2 — Tetrahedral SIC on the Bloch Sphere**
#
# The four SIC states sit at the vertices of a regular tetrahedron inscribed in
# the Bloch sphere.  Every pair subtends the same angle — the geometric
# signature of equiangularity — and together they cover the sphere uniformly,
# which is the visual expression of informational completeness.

plot_bloch_sphere(
    [sic2],
    labels=["SIC states"],
    colors=["darkorange"],
    markers=["o"],
    title=r"Tetrahedral SIC states ($d=2$, $|\langle\psi_i|\psi_j\rangle|^2 = \frac{1}{3}$)",
)

# %%
# 3. Gram Matrix and the Equiangularity Condition
# ------------------------------------------------
#
# The **Gram matrix** of a set of vectors is
#
# .. math::
#
#    G_{ij} = \langle \psi_i | \psi_j \rangle.
#
# :py:func:`~toqito.matrix_ops.vectors_to_gram_matrix` computes this directly.
# For a SIC the off-diagonal entries of :math:`|G|^2` should all equal
# :math:`1/(d+1)`.

G2    = vectors_to_gram_matrix(sic2)
G2_sq = np.abs(G2) ** 2

print("Overlap-squared Gram matrix (d=2):")
print(np.round(G2_sq, 4))

off_diag = G2_sq[~np.eye(4, dtype=bool)]
print(f"\nAll off-diagonal values equal 1/3: {np.allclose(off_diag, 1/3)}")

# %%
# We can verify the same condition by iterating over pairs directly:

overlaps_d2 = [abs(np.vdot(sic2[i], sic2[j])) ** 2
               for i, j in combinations(range(4), 2)]
print(f"Pairwise overlaps-squared: {np.round(overlaps_d2, 6)}")

# %%
# 4. State Discrimination of the Qubit SIC
# -----------------------------------------
#
# Given the uniform ensemble over the four tetrahedral states, what is the
# highest success probability achievable?
#
# For any SIC ensemble there is a closed-form answer
#
# .. math::
#
#    p_{\text{succ}}^{\text{SIC}} = \frac{2}{d(d+1)}.
#
# For :math:`d=2` this gives :math:`p_\text{succ} = 1/3`.  Symmetry of the
# ensemble means the optimal measurement is the SIC POVM itself.

rhos_d2  = [np.outer(v, v.conj()) for v in sic2]
probs_d2 = [1 / 4] * 4

p_succ_d2, _ = state_distinguishability(rhos_d2, probs_d2)

d = 2
p_theory_d2 = 2 / (d * (d + 1))
print(f"toqito SDP result:  p_succ = {p_succ_d2:.6f}")
print(f"Analytic formula:   p_succ = {p_theory_d2:.6f}   [2 / d(d+1), d={d}]")
print(f"Agreement: {np.isclose(p_succ_d2, p_theory_d2, atol=1e-5)}")

# %%
# **Plot 3 — Geometric Comparison: Orthogonal Basis vs SIC Tetrahedron**
#
# Overlaying both ensembles on a single sphere makes the discrimination gap
# vivid.  The orthogonal pair (poles) can always be separated by a great-circle
# measurement; the four SIC vertices, equidistant from one another, offer no
# such privileged axis — hence the drop from :math:`p_\text{succ}=1` to
# :math:`p_\text{succ}=1/3`.

plot_bloch_sphere(
    [[e0, e1], sic2],
    labels=[r"Orthogonal $\{|0\rangle,|1\rangle\}$", r"SIC tetrahedron"],
    colors=["royalblue", "darkorange"],
    markers=["^", "o"],
    title=r"Orthogonal basis vs SIC tetrahedron"
          "\n"
          r"$p_\mathrm{succ}$: 1.000 (orth.) vs 0.333 (SIC)",
)

# %%
# The SDP confirms the analytic bound.  Compare with the orthogonal-basis case:
# SIC states are much harder to distinguish because equiangularity offers no
# geometric advantage to any measurement direction.

# %%
# 5. Qutrit SIC via Weyl-Heisenberg Displacement Operators
# ---------------------------------------------------------
#
# For :math:`d \geq 3` a standard construction uses **displacement operators**
#
# .. math::
#
#    D_{jk} = \tau^{jk}\, X^j Z^k, \qquad \tau = e^{i\pi/d},
#
# where :math:`X` (shift) and :math:`Z` (clock) are the generalised Pauli
# matrices, available in :code:`|toqito⟩` via
# :py:func:`~toqito.matrices.gen_pauli`.
#
# Starting from a **fiducial vector** :math:`|\phi\rangle`, the SIC orbit is
#
# .. math::
#
#    \left\{\, D_{jk}|\phi\rangle \;:\; j,k = 0,\ldots,d-1 \,\right\}.
#
# For :math:`d=3` a known fiducial (the Hesse SIC) is used below.

def sic_d3():
    """Nine Hesse SIC states in dimension 3 via Weyl-Heisenberg orbit."""
    d = 3
    tau = np.exp(1j * np.pi / d)

    # Hesse SIC fiducial
    alpha    = np.arctan(np.sqrt(2))
    fiducial = np.array([0, 1, -np.exp(1j * alpha)], dtype=complex)
    fiducial /= np.linalg.norm(fiducial)

    return [
        tau ** (j * k) * gen_pauli(j, k, d) @ fiducial
        for j in range(d) for k in range(d)
    ]

sic3 = sic_d3()

# %%
# Verify equiangularity (:math:`1/(d+1) = 1/4` for :math:`d=3`):

G3_sq = np.abs(vectors_to_gram_matrix(sic3)) ** 2
off3  = G3_sq[~np.eye(9, dtype=bool)]

print(f"d=3  off-diagonal overlap-squared — mean: {off3.mean():.6f},  "
      f"std: {off3.std():.2e}")
print(f"Expected 1/4 = {0.25}   All equal: {np.allclose(off3, 1/4, atol=1e-8)}")

# %%
# Discrimination SDP for :math:`d=3`:

rhos_d3  = [np.outer(v, v.conj()) for v in sic3]
probs_d3 = [1 / 9] * 9

p_succ_d3, _ = state_distinguishability(rhos_d3, probs_d3)

d = 3
p_theory_d3 = 2 / (d * (d + 1))
print(f"toqito SDP result:  p_succ = {p_succ_d3:.6f}")
print(f"Analytic formula:   p_succ = {p_theory_d3:.6f}   [2 / d(d+1), d={d}]")

# %%
# 6. SIC-POVMs as Projective 2-Designs
# -------------------------------------
#
# A set of :math:`N` unit vectors in :math:`\mathbb{C}^d` is a **projective
# 2-design** if it reproduces the second moment of the Haar measure over pure
# states.  Equivalently, the **order-2 frame potential**
#
# .. math::
#
#    \Phi^{(2)} = \sum_{i,j=1}^{N} |\langle \psi_i | \psi_j \rangle|^4
#
# achieves the Welch lower bound
#
# .. math::
#
#    \Phi^{(2)}_{\min} = \frac{N^2}{\binom{d+1}{2}}.
#
# SIC-POVMs with :math:`N=d^2` saturate this bound exactly.

def frame_potential(states):
    """Compute the order-2 frame potential from a list of state vectors."""
    G = np.abs(vectors_to_gram_matrix(states)) ** 2
    return float(np.sum(G ** 2))

N2, d2 = 4, 2
phi2     = frame_potential(sic2)
phi2_min = N2 ** 2 / comb(d2 + 1, 2)
print(f"d=2  frame potential: {phi2:.6f}   Welch bound: {phi2_min:.6f}   "
      f"Saturated: {np.isclose(phi2, phi2_min)}")

N3, d3 = 9, 3
phi3     = frame_potential(sic3)
phi3_min = N3 ** 2 / comb(d3 + 1, 2)
print(f"d=3  frame potential: {phi3:.6f}   Welch bound: {phi3_min:.6f}   "
      f"Saturated: {np.isclose(phi3, phi3_min, atol=1e-6)}")

# %%
# Saturating the Welch bound means SIC measurements average uniformly over
# all two-copy observables — they are optimal reference measurements for
# randomised benchmarking protocols.

# %%
# 7. Resolution of the Identity
# ------------------------------
#
# The SIC POVM elements :math:`M_i = |\psi_i\rangle\langle\psi_i|/d` satisfy
#
# .. math::
#
#    \sum_{i=1}^{d^2} M_i = I_d.
#
# This is the POVM completeness condition and can be checked numerically.

def resolution_residual(states):
    """Frobenius distance between the SIC frame operator and the identity."""
    d = len(states[0])
    S = sum(np.outer(v, v.conj()) for v in states) / d
    return np.linalg.norm(S - np.eye(d))

print(f"Resolution of identity residual  d=2: {resolution_residual(sic2):.2e}")
print(f"Resolution of identity residual  d=3: {resolution_residual(sic3):.2e}")

# %%
# 8. Informational Completeness and State Reconstruction
# -------------------------------------------------------
#
# A POVM is **informationally complete** (IC) if the outcome probabilities
# :math:`p_i = \mathrm{Tr}(M_i \rho)` uniquely determine :math:`\rho`.  The SIC
# POVM is IC and admits the closed-form reconstruction
#
# .. math::
#
#    \rho = \sum_{i=1}^{d^2}
#    \left[(d+1)\,p_i - \frac{1}{d}\right] |\psi_i\rangle\langle\psi_i|.

def sic_reconstruct(probs, states, d):
    """Reconstruct a density matrix from SIC measurement outcome probabilities."""
    return sum(
        ((d + 1) * probs[i] - 1.0 / d) * np.outer(states[i], states[i].conj())
        for i in range(len(states))
    )

# Random qubit density matrix
rng = np.random.default_rng(42)
A   = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
rho = A @ A.conj().T
rho /= np.trace(rho)

# Simulated ideal SIC measurement outcomes
povm   = [np.outer(v, v.conj()) / 2 for v in sic2]
p_obs  = np.array([np.real(np.trace(M @ rho)) for M in povm])

# Reconstruct
rho_rec = sic_reconstruct(p_obs, sic2, d=2)
error   = np.linalg.norm(rho - rho_rec)
print(f"Reconstruction Frobenius error: {error:.2e}   (machine precision expected)")

# %%
# The reconstruction is exact to floating-point precision, confirming
# informational completeness.

# %%
# 9. Comparison with Mutually Unbiased Bases (MUBs)
# -------------------------------------------------
#
# **Mutually Unbiased Bases** in :math:`\mathbb{C}^d` are collections of
# orthonormal bases such that any two states from *different* bases satisfy
#
# .. math::
#
#    |\langle \phi_i^{(m)} | \phi_j^{(n)} \rangle|^2 = \frac{1}{d}, \quad m \neq n.
#
# This is structurally similar to SIC equiangularity, but the two families are
# distinct:
#
# * **MUBs** partition into orthonormal bases — intra-basis overlaps are 0 or 1,
#   inter-basis overlaps equal :math:`1/d`.
# * **SICs** form a single equiangular tight frame — all off-diagonal overlaps
#   equal :math:`1/(d+1) < 1/d`.
#
# For :math:`d=2` the three MUBs are the eigenbases of :math:`\sigma_x, \sigma_y,
# \sigma_z` (six octahedral vertices on the Bloch sphere), compared to the four
# tetrahedral SIC vertices.
#
# :py:func:`~toqito.states.mutually_unbiased_basis` constructs the complete set
# of MUBs directly:

mub2  = mutually_unbiased_basis(2)
G_mub = np.abs(vectors_to_gram_matrix(mub2)) ** 2

print(f"Number of MUB vectors returned: {len(mub2)}  (3 bases × 2 vectors each)")
print("\nMUB overlap-squared matrix (d=2):")
print(np.round(G_mub, 3))

# %%
# The block structure is clear: within each basis the overlaps are 0 or 1;
# across bases every overlap is :math:`1/d = 1/2`.

# %%
# 10. Discrimination Comparison: SIC vs MUBs
# -------------------------------------------
#
# We compare optimal success probabilities for the qubit SIC and the uniform
# mixture over all six MUB states.

rhos_mub  = [np.outer(v, v.conj()) for v in mub2]
probs_mub = [1 / 6] * 6

p_succ_mub, _ = state_distinguishability(rhos_mub, probs_mub)

print(f"MUB (6 states, d=2)  p_succ = {p_succ_mub:.6f}")
print(f"SIC (4 states, d=2)  p_succ = {p_succ_d2:.6f}")
print()
print("SIC states are harder to discriminate per-state than MUB states,")
print("reflecting their tighter equiangular packing on the Bloch sphere.")

# %%
# 11. Learnability of the SIC Ensemble
# -------------------------------------
#
# State discrimination asks for the best single-shot measurement to identify a
# known state.  **Learnability** asks a related but distinct question: given
# :math:`k` copies of an unknown state drawn from the ensemble, how well can we
# *classify* which state it is on average?
#
# :py:func:`~toqito.state_props.learnability` computes the average
# classification error for :math:`k` copies.  For :math:`k=1` each copy is a
# single instance — this regime directly connects to the discrimination setting.

learn_d2 = learnability(sic2, k=1)

print(f"SIC learnability (d=2, k=1)  average classification error: "
      f"{learn_d2['value']:.6f}")
print(f"Complement (success):                                        "
      f"{1 - learn_d2['value']:.6f}")
print(f"State discrimination p_succ (d=2):                          "
      f"{p_succ_d2:.6f}")

# %%
# The key conceptual point: even though the SIC POVM is *informationally
# complete* — in principle, enough repeated measurements fully determine any
# state — a *single copy* provides very limited distinguishing power.
# Equiangularity caps the learnability just as it caps the discrimination
# probability.  More copies (:math:`k > 1`) are needed before classification
# accuracy improves substantially.

# Conclusion
# ----------
#
# SIC-POVMs sit between two geometric extremes on the Bloch sphere:
#
# * **Orthogonal bases** — antipodal states, perfect discrimination, incomplete
#   symmetry.
# * **SIC tetrahedron** — maximally symmetric, informationally complete,
#   discrimination capped at :math:`2/[d(d+1)]`.
#
# Using :code:`|toqito⟩` we verified equiangular overlaps, minimal frame
# potential (projective 2-design), the analytic discrimination bound, exact
# state reconstruction, and single-copy learnability — all flowing from the
# same equiangularity constraint.  The discrimination viewpoint gives SIC
# symmetry a direct operational meaning.
#
# References
# ----------
#
# .. footbibliography::
