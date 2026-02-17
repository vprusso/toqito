"""SIC-POVMs: Geometry, Structure, and Quantum State Discrimination
====================================================================

In this tutorial, we explore **Symmetric Informationally Complete
Positive Operator-Valued Measures** (SIC-POVMs). We use :code:`|toqito⟩`
to examine their geometric properties and apply them to quantum state
discrimination (QSD).

"""

# %%
# A SIC-POVM in dimension :math:`d` is a set of :math:`d^2` unit vectors
# :math:`\\{|\\psi_i\\rangle\\}_{i=1}^{d^2} \\subset \\mathbb{C}^d` satisfying
# the **symmetric inner product condition**:
#
# .. math::
#    |\\langle \\psi_i | \\psi_j \\rangle|^2 = \\frac{1}{d+1}, \\quad \\forall \\, i \\neq j
#
# The associated POVM elements are:
#
# .. math::
#    M_i = \\frac{1}{d} |\\psi_i\\rangle\\langle\\psi_i|
#
# This ensures :math:`\\sum_{i=1}^{d^2} M_i = I_d` (completeness), and the
# equal-overlap condition makes SIC-POVMs **informationally complete** 
# allowing reconstruction of any quantum state from measurement statistics.
#
# :footcite:`Renes_2004_Symmetric,Scott_2010_SIC`.
#
# The d=2 SIC-POVM: Trine and Tetrahedron
# -----------------------------------------
#
# The simplest SIC-POVM lives in :math:`d=2`, with :math:`d^2 = 4` elements.
#  **trine** — three states equally spaced on the
# equator of the Bloch sphere. The canonical SIC-POVM in :math:`d=2` consists
# of **4 states** forming a regular tetrahedron on the Bloch sphere, with
# equal-overlap condition :math:`|\\langle\\psi_i|\\psi_j\\rangle|^2 = 1/3`
# for all :math:`i \\neq j`.

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from toqito.states import trine
from toqito.state_opt import state_distinguishability, state_exclusion

np.set_printoptions(precision=4, suppress=True)

trine_states = trine()
t1, t2, t3 = trine_states[:, 0], trine_states[:, 1], trine_states[:, 2]

# Verify trine overlap: |<ti|tj>|^2 = 1/3 for all i != j
assert all(
    np.isclose(abs(np.vdot(a, b)) ** 2, 1 / 3, atol=1e-10)
    for a, b in [(t1, t2), (t1, t3), (t2, t3)]
)


# %%
# We now construct the canonical d=2 SIC-POVM (tetrahedral states) and verify
# the symmetric overlap condition holds for all pairs.


def sic_povm_d2() -> list[np.ndarray]:
    """Return the 4 SIC-POVM states for d=2 (tetrahedral states on Bloch sphere)."""
    psi_0 = np.array([1, 0], dtype=complex)
    psi_1 = np.array([1 / np.sqrt(3), np.sqrt(2 / 3)], dtype=complex)
    psi_2 = np.array([1 / np.sqrt(3), np.sqrt(2 / 3) * np.exp(1j * 2 * np.pi / 3)], dtype=complex)
    psi_3 = np.array([1 / np.sqrt(3), np.sqrt(2 / 3) * np.exp(1j * 4 * np.pi / 3)], dtype=complex)
    return [psi_0, psi_1, psi_2, psi_3]


sic2_states = sic_povm_d2()
d = 2

# Verify all pairwise SIC overlaps equal 1/(d+1)
assert all(
    np.isclose(abs(np.vdot(sic2_states[i], sic2_states[j])) ** 2, 1 / (d + 1), atol=1e-10)
    for i, j in combinations(range(len(sic2_states)), 2)
)


# %%
# We now build the POVM elements :math:`M_i = \\frac{1}{d}|\\psi_i\\rangle\\langle\\psi_i|`
# and verify completeness, i.e. that :math:`\\sum_i M_i = I_d`.


def build_povm(states: list[np.ndarray], d: int) -> list[np.ndarray]:
    """Build POVM elements M_i = (1/d)|psi_i><psi_i| from SIC states."""
    return [np.outer(psi, psi.conj()) / d for psi in states]


povm2 = build_povm(sic2_states, d=2)

# Verify completeness: ||sum(M_i) - I|| ~ 0
assert np.linalg.norm(sum(povm2) - np.eye(2)) < 1e-10


# %%
# Bloch Sphere Visualization
# ---------------------------
#
# The four SIC-POVM states form a regular tetrahedron on the Bloch sphere.
# We visualize the Bloch vectors :math:`\\vec{r}_i` where
# :math:`\\rho_i = \\frac{1}{2}(I + \\vec{r}_i \\cdot \\vec{\\sigma})`.


def state_to_bloch(psi: np.ndarray) -> np.ndarray:
    """Convert a qubit pure state |psi> to its Bloch vector (rx, ry, rz)."""
    rho = np.outer(psi, psi.conj())
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.array([np.trace(rho @ s).real for s in [sx, sy, sz]])


bloch_vecs = [state_to_bloch(psi) for psi in sic2_states]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
ax.plot_wireframe(
    np.outer(np.cos(u), np.sin(v)),
    np.outer(np.sin(u), np.sin(v)),
    np.outer(np.ones(np.size(u)), np.cos(v)),
    color="lightgray", alpha=0.2, linewidth=0.5,
)

colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]
labels = [r"$|\psi_0\rangle$", r"$|\psi_1\rangle$", r"$|\psi_2\rangle$", r"$|\psi_3\rangle$"]

for bv, col, lab in zip(bloch_vecs, colors, labels):
    ax.scatter(*bv, color=col, s=120, zorder=5, label=lab)
    ax.quiver(0, 0, 0, *bv, color=col, alpha=0.7, arrow_length_ratio=0.1)
    ax.text(*(bv * 1.12), lab, fontsize=11, ha="center")

for i, j in combinations(range(4), 2):
    pts = np.array([bloch_vecs[i], bloch_vecs[j]]).T
    ax.plot(*pts, "k--", alpha=0.3, linewidth=1)

for vec, lbl in [([1.3, 0, 0], "x"), ([0, 1.3, 0], "y"), ([0, 0, 1.3], "z")]:
    ax.quiver(0, 0, 0, *vec, color="black", alpha=0.5, arrow_length_ratio=0.08)
    ax.text(*[v * 1.05 for v in vec], lbl, fontsize=10)

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_zlim(-1.2, 1.2)
ax.set_title("d=2 SIC-POVM: Tetrahedron on Bloch Sphere", fontsize=13, pad=15)
ax.legend(loc="upper left", fontsize=9)
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.savefig("sic_bloch.png", dpi=120, bbox_inches="tight")
plt.show()


# %%
# SIC-POVM in Dimension 3
# ------------------------
#
# In :math:`d=3`, a SIC-POVM consists of **9 states** with pairwise overlaps
# :math:`|\\langle\\psi_i|\\psi_j\\rangle|^2 = 1/4` for :math:`i \\neq j`.
# A known analytic solution (Hesse SIC) is generated via a Weyl-Heisenberg
# group orbit :footcite:`Renes_2004_Symmetric`.
#
# The displacement operators are:
#
# .. math::
#    D_{jk} = \\tau^{jk} X^j Z^k
#
# where :math:`X|m\\rangle = |m+1 \\bmod d\\rangle`,
# :math:`Z|m\\rangle = \\omega^m|m\\rangle`,
# :math:`\\omega = e^{2\\pi i/d}`, and :math:`\\tau = e^{\\pi i/d}`.


def weyl_heisenberg_sic_d3() -> list[np.ndarray]:
    """Generate the d=3 Hesse SIC-POVM via Weyl-Heisenberg displacement operators."""
    d = 3
    omega = np.exp(2j * np.pi / d)
    tau = np.exp(1j * np.pi / d)
    X = np.roll(np.eye(d), -1, axis=0)
    Z = np.diag([omega**m for m in range(d)])

    t = np.exp(1j * np.arctan(np.sqrt(2)))
    fiducial = np.array([0, 1, -t], dtype=complex)
    fiducial /= np.linalg.norm(fiducial)

    states = []
    for j in range(d):
        for k in range(d):
            D_jk = tau ** (j * k) * (np.linalg.matrix_power(X, j) @ np.linalg.matrix_power(Z, k))
            states.append(D_jk @ fiducial)
    return states


sic3_states = weyl_heisenberg_sic_d3()
d = 3

# Verify all pairwise overlaps equal 1/(d+1) = 1/4
overlaps = [
    abs(np.vdot(sic3_states[i], sic3_states[j])) ** 2
    for i, j in combinations(range(len(sic3_states)), 2)
]
assert np.allclose(overlaps, 1 / (d + 1), atol=1e-8)

povm3 = build_povm(sic3_states, d=3)
assert np.linalg.norm(sum(povm3) - np.eye(3)) < 1e-10


# %%
# Gram Matrix and Geometric Structure
# -------------------------------------
#
# A clean signature of SIC-POVMs is their **Gram matrix**
# :math:`G_{ij} = |\\langle\\psi_i|\\psi_j\\rangle|^2`. For a :math:`d`-dimensional
# SIC-POVM, this is a :math:`d^2 \\times d^2` matrix with 1s on the diagonal
# and :math:`1/(d+1)` everywhere else.


def gram_matrix(states: list[np.ndarray]) -> np.ndarray:
    """Compute the Gram matrix G_ij = |<psi_i|psi_j>|^2."""
    n = len(states)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = abs(np.vdot(states[i], states[j])) ** 2
    return G


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, states, title in [
    (axes[0], sic2_states, "d=2 SIC Gram Matrix (4×4)"),
    (axes[1], sic3_states, "d=3 SIC Gram Matrix (9×9)"),
]:
    G = gram_matrix(states)
    im = ax.imshow(G, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("State index j")
    ax.set_ylabel("State index i")
    n = len(states)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    plt.colorbar(im, ax=ax, label=r"$|\langle\psi_i|\psi_j\rangle|^2$")

plt.suptitle("SIC-POVM Gram Matrices: Uniform Off-Diagonal Structure", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("sic_gram.png", dpi=120, bbox_inches="tight")
plt.show()


# %%
# Quantum State Discrimination with SIC-POVMs
# ---------------------------------------------
#
# Minimum-Error Discrimination
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Given the :math:`d^2` SIC-POVM states prepared with equal prior probability
# :math:`p_i = 1/d^2`, the optimal **minimum-error** success probability is
# known analytically :footcite:`Renes_2004_Symmetric`:
#
# .. math::
#    P_{\\text{succ}} = \\frac{2}{d(d+1)}
#
# We verify this using :py:func:`~toqito.state_opt.state_distinguishability`.


def states_to_density_matrices(states: list[np.ndarray]) -> list[np.ndarray]:
    """Convert list of state vectors to density matrices."""
    return [np.outer(psi, psi.conj()) for psi in states]


rho_list_2 = states_to_density_matrices(sic2_states)
priors_2 = [1 / len(rho_list_2)] * len(rho_list_2)
val_2, _ = state_distinguishability(rho_list_2, priors_2)
assert np.isclose(val_2, 2 / (2 * 3), atol=1e-4)  # d=2: 2/(d*(d+1)) = 1/3

rho_list_3 = states_to_density_matrices(sic3_states)
priors_3 = [1 / len(rho_list_3)] * len(rho_list_3)
val_3, _ = state_distinguishability(rho_list_3, priors_3)
assert np.isclose(val_3, 2 / (3 * 4), atol=1e-4)  # d=3: 2/(d*(d+1)) = 1/6


# %%
# State Exclusion
# ~~~~~~~~~~~~~~~~
#
# In state exclusion (antidistinguishability), we ask which state was *not*
# prepared. We compute both discrimination and exclusion for the d=2 SIC-POVM
# using :py:func:`~toqito.state_opt.state_exclusion`.

excl_val, _ = state_exclusion(rho_list_2, priors_2)


# %%
# Discrimination as a Function of Dimension
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The analytic formula :math:`P_{\\text{succ}} = \\frac{2}{d(d+1)}` shows that
# SIC-POVM state discrimination becomes **harder in higher dimensions** — more
# states to distinguish, each pair more similar.

dims = range(2, 9)
p_analytic = [2 / (d * (d + 1)) for d in dims]
p_random = [1 / d**2 for d in dims]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(dims, p_analytic, "o-", color="#E63946", linewidth=2, markersize=8, label=r"SIC optimal: $\frac{2}{d(d+1)}$")
ax.plot(dims, p_random, "s--", color="#457B9D", linewidth=2, markersize=8, label=r"Random guess: $\frac{1}{d^2}$")
ax.set_xlabel("Dimension d", fontsize=12)
ax.set_ylabel("Success probability", fontsize=12)
ax.set_title("SIC-POVM Min-Error Discrimination vs. Dimension", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(list(dims))
plt.tight_layout()
plt.savefig("sic_dim_scaling.png", dpi=120, bbox_inches="tight")
plt.show()


# %%
# Informational Completeness
# ---------------------------
#
# A key property of SIC-POVMs is **informational completeness**: the
# measurement statistics :math:`p_i = \\text{Tr}(M_i \\rho)` uniquely determine
# :math:`\\rho`. Any quantum state can be **reconstructed** from SIC-POVM
# measurement outcomes via:
#
# .. math::
#    \\rho = \\sum_{i=1}^{d^2} \\left[(d+1) p_i - \\frac{1}{d}\\right] |\\psi_i\\rangle\\langle\\psi_i|
#
# where :math:`p_i = \\text{Tr}(M_i \\rho)`.


def sic_reconstruction(p: np.ndarray, states: list[np.ndarray], d: int) -> np.ndarray:
    """Reconstruct a density matrix from SIC-POVM probabilities."""
    rho_rec = np.zeros((d, d), dtype=complex)
    for i, psi in enumerate(states):
        rho_rec += ((d + 1) * p[i] - 1 / d) * np.outer(psi, psi.conj())
    return rho_rec


def get_sic_probs(rho: np.ndarray, povm: list[np.ndarray]) -> np.ndarray:
    """Compute SIC-POVM measurement probabilities p_i = Tr(M_i rho)."""
    return np.array([np.trace(M @ rho).real for M in povm])


np.random.seed(42)
A = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
rho_test = A @ A.conj().T
rho_test /= np.trace(rho_test)

p2 = get_sic_probs(rho_test, povm2)
rho_rec = sic_reconstruction(p2, sic2_states, d=2)

# Verify reconstruction is exact
assert np.linalg.norm(rho_test - rho_rec) < 1e-10


# %%
# Rank of the Measurement Frame
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Informational completeness is equivalent to the :math:`d^2` rank-1 operators
# :math:`\\{|\\psi_i\\rangle\\langle\\psi_i|\\}` **spanning** the space of Hermitian
# :math:`d \\times d` matrices (which has dimension :math:`d^2`).


def frame_operator_rank(states: list[np.ndarray]) -> int:
    """Check spanning rank by vectorizing |psi_i><psi_i| and computing matrix rank."""
    vecs = [np.outer(psi, psi.conj()).ravel() for psi in states]
    return np.linalg.matrix_rank(np.array(vecs), tol=1e-8)


for d, states in [(2, sic2_states), (3, sic3_states)]:
    assert frame_operator_rank(states) == d**2  # d^2 rank => informationally complete


# %%
# Group Covariance and Weyl-Heisenberg Structure
# -----------------------------------------------
#
# Most known SIC-POVMs are **group-covariant**: they arise as orbits of a
# fiducial state under the Weyl-Heisenberg group :footcite:`Appleby_2005_SIC`.
# The displacement operators :math:`D_{jk} = \\tau^{jk} X^j Z^k` generate
# :math:`d^2` states from a single **fiducial** :math:`|\\psi_0\\rangle`.
# We verify the algebra satisfies
# :math:`\\text{Tr}(D_{jk}^\\dagger D_{j'k'}) = d \\, \\delta_{jj'} \\delta_{kk'}`.


def displacement_operators(d: int) -> list[tuple[int, int, np.ndarray]]:
    """Generate all d^2 Weyl-Heisenberg displacement operators for dimension d."""
    omega = np.exp(2j * np.pi / d)
    tau = np.exp(1j * np.pi / d)
    X = np.roll(np.eye(d), -1, axis=0)
    Z = np.diag([omega**m for m in range(d)])
    return [
        (j, k, tau ** (j * k) * (np.linalg.matrix_power(X, j) @ np.linalg.matrix_power(Z, k)))
        for j in range(d)
        for k in range(d)
    ]


d = 3
disps = displacement_operators(d)

# Verify unitarity and orthogonality
assert all(np.allclose(D @ D.conj().T, np.eye(d), atol=1e-10) for _, _, D in disps)
assert all(
    np.isclose(np.trace(D1.conj().T @ D2), d if (j1 == j2 and k1 == k2) else 0, atol=1e-9)
    for j1, k1, D1 in disps
    for j2, k2, D2 in disps
)


# %%
# Comparison with MUBs
# ---------------------
#
# How do SIC-POVMs compare to Mutually Unbiased Bases (MUBs) in terms of
# discrimination power? For :math:`d=2`, the 6 eigenstates of the Pauli
# operators X, Y, Z form 3 MUBs. We compare their optimal discrimination
# probabilities using :py:func:`~toqito.state_opt.state_distinguishability`.


def mub_states_d2() -> list[np.ndarray]:
    """The 6 states from 3 MUBs in d=2 (eigenstates of X, Y, Z Pauli operators)."""
    return [
        np.array([1, 0], dtype=complex),
        np.array([0, 1], dtype=complex),
        np.array([1, 1], dtype=complex) / np.sqrt(2),
        np.array([1, -1], dtype=complex) / np.sqrt(2),
        np.array([1, 1j], dtype=complex) / np.sqrt(2),
        np.array([1, -1j], dtype=complex) / np.sqrt(2),
    ]


mub_rhos = states_to_density_matrices(mub_states_d2())
mub_priors = [1 / len(mub_rhos)] * len(mub_rhos)
val_mub, _ = state_distinguishability(mub_rhos, mub_priors)

# Both SIC-POVM and MUBs exceed random guessing
assert val_2 > 1 / len(rho_list_2)
assert val_mub > 1 / len(mub_rhos)


# %%
# Interpreting the Results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Through this tutorial, we explored SIC-POVMs using :code:`|toqito⟩`:
#
# **Geometric structure:** SIC-POVM states are maximally spread — equal
# pairwise overlaps :math:`|\\langle\\psi_i|\\psi_j\\rangle|^2 = 1/(d+1)` — forming
# regular simplices in projective Hilbert space. In :math:`d=2`, they inscribe a
# tetrahedron on the Bloch sphere.
#
# **Informational completeness:** The :math:`d^2` SIC states span all Hermitian
# matrices on :math:`\\mathbb{C}^d`, enabling perfect quantum state tomography
# from measurement statistics via
# :math:`\\rho = \\sum_i [(d+1)p_i - 1/d]|\\psi_i\\rangle\\langle\\psi_i|`.
#
# **Discrimination hardness:** For equal priors, the optimal success probability
# is :math:`P_{\\text{succ}} = 2/[d(d+1)]`, which decreases as dimension grows —
# more states, more similar to each other.
#
# **Group covariance:** Most known SIC-POVMs arise as Weyl-Heisenberg orbits of
# a fiducial state, revealing a deep connection between symmetric measurements
# and the structure of the displacement operator group.
#
# :footcite:`Zauner_1999_Quantum`.

# %%
#
# References
# ----------
#
# .. footbibliography::
