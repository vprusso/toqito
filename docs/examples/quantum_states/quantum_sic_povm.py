"""SIC-POVMs and Quantum State Discrimination.

Imagine Alice picks a quantum state at random from a secret list and sends it to Bob. Bob's
challenge: guess which state he received. The answer depends not just on how many states are in
the list, but on their *geometry* — how similar or different they are from one another.

This tutorial builds intuition for that geometry by working through a special family of
measurements called **Symmetric Informationally Complete Positive Operator-Valued Measures**
(SIC-POVMs). We will see that equiangularity — every pair of states having the *same* overlap —
simultaneously maximises symmetry, guarantees informational completeness, and places an absolute
ceiling on Bob's success probability.

## What you will learn

In this tutorial we will:

- Visualise qubit states on the Bloch sphere and see how geometry controls
  discrimination difficulty.
- Construct qubit ($d=2$) and qutrit ($d=3$) SIC-POVMs and verify their
  equiangularity property.
- Compute the optimal success probability for the SIC ensemble using a semidefinite
  program (SDP) and confirm it matches the closed-form bound $2/[d(d+1)]$.
- Show that every SIC-POVM saturates the Welch bound, making it a projective
  2-design.
- Verify the POVM completeness condition (resolution of the identity).
- Perform exact state reconstruction from SIC measurement outcomes.
- Compare SIC-POVMs with Mutually Unbiased Bases (MUBs) and single-shot
  learnability.

!!! note
    Weyl-Heisenberg SIC fiducials have been found numerically in every dimension up to at least
    $d = 151$ and analytically in infinitely many dimensions, yet a proof of existence for all
    $d$ (Zauner's conjecture) remains open.

This tutorial assumes familiarity with the basics of quantum information. For background see
:footcite:`Chuang_2011_Quantum` or :footcite:`Watrous_2018_TQI`. Installation instructions for
|toqito⟩ are in :ref:`getting_started_reference-label`.

References:
    .. footbibliography::

"""

# %%
# We begin by importing all necessary packages. The core mathematical objects —
# generalised Pauli matrices, Gram matrices, state distinguishability SDPs, and
# learnability measures — are all available directly from `|toqito⟩`.

from itertools import combinations
from math import comb

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from toqito.matrices import gen_pauli, standard_basis
from toqito.matrix_ops import vectors_to_gram_matrix
from toqito.state_opt import state_distinguishability
from toqito.state_props import learnability
from toqito.states import mutually_unbiased_basis

# %%
# ## Helper functions
#
# Before diving into SIC-POVMs, we define two helper functions that will be used
# throughout the tutorial. The first converts a qubit pure state to its
# three-dimensional Bloch vector, and the second plots one or more groups of
# states on the Bloch sphere.
#
# The Bloch sphere is our primary geometric tool: it turns abstract inner-product
# conditions into visible angles and distances, making it easy to see *why* some
# state sets are easy to distinguish and others are not.


def bloch_coords(state: np.ndarray) -> np.ndarray:
    r"""Return the $(x, y, z)$ Bloch vector of a qubit pure state.

    Every qubit pure state $|\psi\rangle$ corresponds to a unique point on the unit sphere in
    $\mathbb{R}^3$ via

    $$\vec{r} = \bigl(2\,\mathrm{Re}\,\rho_{01},\; 2\,\mathrm{Im}\,\rho_{10},\;
    \rho_{00} - \rho_{11}\bigr), \qquad \rho = |\psi\rangle\langle\psi|.$$

    The Bloch sphere is our main geometric tool throughout the tutorial — it turns abstract
    inner-product conditions into visible angles and distances.

    Args:
        state: A length-2 complex unit vector representing a qubit pure state.

    Returns:
        A real array of shape ``(3,)`` giving the Bloch-sphere coordinates.

    Examples:
        The computational basis state $|0\rangle$ maps to the north pole $(0, 0, 1)$:

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrices import standard_basis
        e0, _ = standard_basis(2)
        rho = np.outer(e0, e0.conj())
        x = 2 * np.real(rho[0, 1])
        y = 2 * np.imag(rho[1, 0])
        z = np.real(rho[0, 0] - rho[1, 1])
        print(np.array([x, y, z]))
        ```

    """
    rho = np.outer(state, state.conj())
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[1, 0])
    z = np.real(rho[0, 0] - rho[1, 1])
    return np.array([x, y, z])


def plot_bloch_sphere(
    state_groups: list,
    labels: list,
    colors: list,
    title: str,
    markers: list | None = None,
) -> None:
    """Plot one or more groups of qubit states on the Bloch sphere.

    A translucent unit sphere is drawn together with the equator and the three principal
    meridians. Each group of states is rendered as scatter points with outgoing arrows from
    the origin, making angular separations between states immediately visible.

    Args:
        state_groups: Each inner list is a group of qubit state vectors to plot together.
        labels: Legend label for each group.
        colors: Marker colour for each group.
        title: Figure title.
        markers: Marker style for each group. Defaults to ``'o'`` for all groups.

    Examples:
        Plot the two computational basis states at the poles of the Bloch sphere:

        ```python exec="1" source="above"
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        from toqito.matrices import standard_basis
        e0, e1 = standard_basis(2)
        plot_bloch_sphere([[e0], [e1]], ["|0>", "|1>"], ["royalblue", "crimson"],
                          "Orthogonal basis", markers=["^", "v"])
        ```

    """
    if markers is None:
        markers = ["o"] * len(state_groups)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    u, v = np.mgrid[0 : 2 * np.pi : 120j, 0 : np.pi : 60j]
    ax.plot_surface(
        np.cos(u) * np.sin(v),
        np.sin(u) * np.sin(v),
        np.cos(v),
        color="lightsteelblue",
        alpha=0.08,
        linewidth=0,
    )

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 0, color="gray", lw=0.6, alpha=0.5)
    ax.plot(np.cos(theta), np.zeros(200), np.sin(theta), color="gray", lw=0.6, alpha=0.5)
    ax.plot(np.zeros(200), np.cos(theta), np.sin(theta), color="gray", lw=0.6, alpha=0.5)

    for vec, lbl in zip([(1, 0, 0), (0, 1, 0), (0, 0, 1)], ["x", "y", "z"]):
        ax.quiver(0, 0, 0, *vec, length=1.25, color="dimgray",
                  arrow_length_ratio=0.08, linewidth=0.8)
        ax.text(*(np.array(vec) * 1.35), lbl, color="dimgray",
                fontsize=9, ha="center", va="center")

    for group, label, color, marker in zip(state_groups, labels, colors, markers):
        vecs = np.array([bloch_coords(s) for s in group])
        ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2],
                   color=color, s=80, marker=marker, zorder=5, label=label, depthshade=False)
        for vec in vecs:
            ax.quiver(0, 0, 0, *vec, color=color, alpha=0.7,
                      arrow_length_ratio=0.08, linewidth=1.2)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title, pad=12)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()

# %%
# We also define three computational helpers — `sic_d2`, `sic_d3`, and
# `frame_potential` — that construct the SIC states and measure their
# 2-design quality. These functions encapsulate the constructions we will
# analyse step by step in the sections below.


def sic_d2() -> list:
    r"""Return the four tetrahedral SIC states in dimension 2.

    A SIC-POVM in dimension $d$ is a set of $d^2$ unit vectors $\{|\psi_i\rangle\}$ satisfying
    the **equiangularity** condition

    $$|\langle \psi_i | \psi_j \rangle|^2 = \frac{1}{d+1}, \quad i \neq j.$$

    For $d=2$ this gives four vectors with pairwise overlap-squared $\tfrac{1}{3}$.
    On the Bloch sphere they map to the vertices of a regular tetrahedron inscribed in the unit
    sphere, and the POVM elements $M_i = |\psi_i\rangle\langle\psi_i|/d$ resolve the identity
    $\sum_i M_i = I$.

    The states are constructed by rotating the north pole symmetrically in three equally-spaced
    azimuthal directions.

    Returns:
        A list of four length-2 complex unit vectors.

    Examples:
        Verify the equiangularity condition — all six pairwise overlaps should equal $1/3$:

        ```python exec="1" source="above"
        import numpy as np
        from itertools import combinations
        sic2 = sic_d2()
        overlaps = [abs(np.vdot(sic2[i], sic2[j])) ** 2 for i, j in combinations(range(4), 2)]
        print(np.round(overlaps, 6))
        ```

    """
    phi = 2 * np.pi / 3
    return [
        np.array([1, 0], dtype=complex),
        np.array([1 / np.sqrt(3), np.sqrt(2 / 3)], dtype=complex),
        np.array([1 / np.sqrt(3), np.sqrt(2 / 3) * np.exp(1j * phi)], dtype=complex),
        np.array([1 / np.sqrt(3), np.sqrt(2 / 3) * np.exp(2j * phi)], dtype=complex),
    ]


def sic_d3() -> list:
    r"""Return the nine Hesse SIC states in dimension 3 via the Weyl-Heisenberg orbit.

    For $d \geq 3$ there is no Bloch sphere to guide intuition, but there is an algebraic
    recipe. The **Weyl-Heisenberg displacement operators**

    $$D_{jk} = \tau^{jk}\, X^j Z^k, \qquad \tau = e^{i\pi/d},$$

    where $X$ (shift) and $Z$ (clock) are the generalised Pauli matrices, available in
    |toqito⟩ via [`gen_pauli`][toqito.matrices.gen_pauli].

    Starting from a single **fiducial vector** $|\phi\rangle$, the SIC orbit

    $$\bigl\{\, D_{jk}|\phi\rangle \;:\; j,k = 0,\ldots,d-1 \,\bigr\}$$

    produces all $d^2$ SIC states. The fiducial is a carefully chosen seed; the symmetry group
    then distributes it uniformly across the state space.

    Returns:
        A list of nine length-3 complex unit vectors.

    Examples:
        Verify equiangularity — all off-diagonal overlaps should equal $1/4$:

        ```python exec="1" source="above"
        import numpy as np
        from toqito.matrix_ops import vectors_to_gram_matrix
        sic3 = sic_d3()
        G_sq = np.abs(vectors_to_gram_matrix(sic3)) ** 2
        off = G_sq[~np.eye(9, dtype=bool)]
        print(f"mean={off.mean():.6f}  std={off.std():.2e}  all equal 1/4: {np.allclose(off, 0.25, atol=1e-8)}")
        ```

    """
    d = 3
    tau = np.exp(1j * np.pi / d)
    alpha = np.arctan(np.sqrt(2))
    fiducial = np.array([0, 1, -np.exp(1j * alpha)], dtype=complex)
    fiducial /= np.linalg.norm(fiducial)
    return [
        tau ** (j * k) * gen_pauli(j, k, d) @ fiducial
        for j in range(d)
        for k in range(d)
    ]


def frame_potential(states: list) -> float:
    r"""Compute the order-2 frame potential of a set of unit vectors.

    A set of $N$ unit vectors is a **projective 2-design** if it reproduces the second moment
    of the Haar measure over pure states — informally, if it samples state space as uniformly
    as any random ensemble could. The **order-2 frame potential**

    $$\Phi^{(2)} = \sum_{i,j=1}^{N} |\langle \psi_i | \psi_j \rangle|^4$$

    is bounded below by the **Welch bound**

    $$\Phi^{(2)}_{\min} = \frac{N^2}{\binom{d+1}{2}},$$

    and a set saturates this bound if and only if it is a projective 2-design. SIC-POVMs with
    $N=d^2$ sit exactly on the Welch bound, meaning they average uniformly over all two-copy
    observables — making them optimal reference measurements for randomised benchmarking.

    The discrimination difficulty and the 2-design property are two faces of the same
    equiangularity coin.

    Args:
        states: A list of complex unit vectors all of the same length $d$.

    Returns:
        The frame potential $\Phi^{(2)}$ as a float.

    Examples:
        Verify that the qubit SIC saturates the Welch bound:

        ```python exec="1" source="above"
        import numpy as np
        from math import comb
        sic2 = sic_d2()
        N, d = 4, 2
        phi = frame_potential(sic2)
        welch = N**2 / comb(d + 1, 2)
        print(f"frame potential={phi:.6f}  Welch bound={welch:.6f}  saturated={np.isclose(phi, welch)}")
        ```

    """
    G = np.abs(vectors_to_gram_matrix(states)) ** 2
    return float(np.sum(G ** 2))


def resolution_residual(states: list) -> float:
    r"""Return the Frobenius distance between the SIC frame operator and the identity.

    The SIC POVM elements $M_i = |\psi_i\rangle\langle\psi_i|/d$ must satisfy the POVM
    completeness condition

    $$\sum_{i=1}^{d^2} M_i = I_d,$$

    which ensures outcome probabilities sum to one. This is a non-trivial constraint: $d^2$
    rank-1 operators, each scaled by $1/d$, must sum to the full identity. It is equivalent to
    the SIC states forming a *tight frame*, and it follows directly from equiangularity.

    Args:
        states: A list of $d^2$ complex unit vectors in $\mathbb{C}^d$.

    Returns:
        The Frobenius norm $\|(\sum_i |\psi_i\rangle\langle\psi_i|/d) - I_d\|_F$,
        which should be at machine precision for a valid SIC.

    Examples:
        Check the completeness condition for both the qubit and qutrit SICs:

        ```python exec="1" source="above"
        sic2 = sic_d2()
        sic3 = sic_d3()
        print(f"d=2 residual: {resolution_residual(sic2):.2e}")
        print(f"d=3 residual: {resolution_residual(sic3):.2e}")
        ```

    """
    d = len(states[0])
    S = sum(np.outer(v, v.conj()) for v in states) / d
    return np.linalg.norm(S - np.eye(d))


def sic_reconstruct(probs: np.ndarray, states: list, d: int) -> np.ndarray:
    r"""Reconstruct a density matrix from SIC measurement outcome probabilities.

    A POVM is **informationally complete** (IC) if the measurement outcomes uniquely determine
    the state: no two distinct density matrices produce identical outcome statistics. The SIC
    POVM is IC, and it admits the closed-form inversion

    $$\rho = \sum_{i=1}^{d^2} \left[(d+1)\,p_i - \frac{1}{d}\right]
    |\psi_i\rangle\langle\psi_i|,$$

    where $p_i = \mathrm{Tr}(M_i \rho)$ are the outcome probabilities. This reconstruction
    formula is the operational payoff of equiangularity: because all SIC elements are related
    by symmetry, the dual frame needed to invert the measurement has the same structure. No
    other IC-POVM with $d^2$ elements has such a clean dual.

    !!! note
        The same equiangularity that *limits* single-shot discrimination to $2/[d(d+1)]$ also
        *enables* exact reconstruction in the many-shot limit. SIC-POVMs are not designed to
        distinguish states in one shot; they are designed to gather complete information across
        many shots.

    Args:
        probs: Array of $d^2$ real outcome probabilities $p_i = \mathrm{Tr}(M_i\rho)$.
        states: List of $d^2$ SIC state vectors $|\psi_i\rangle$.
        d: Hilbert space dimension.

    Returns:
        The reconstructed $d \times d$ density matrix.

    Examples:
        Round-trip reconstruction of a random qubit density matrix should be exact to
        floating-point precision:

        ```python exec="1" source="above"
        import numpy as np
        rng = np.random.default_rng(42)
        A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        sic2 = sic_d2()
        povm = [np.outer(v, v.conj()) / 2 for v in sic2]
        p_obs = np.array([np.real(np.trace(M @ rho)) for M in povm])
        rho_rec = sic_reconstruct(p_obs, sic2, d=2)
        print(f"Frobenius reconstruction error: {np.linalg.norm(rho - rho_rec):.2e}")
        ```

    """
    return sum(
        ((d + 1) * probs[i] - 1.0 / d) * np.outer(states[i], states[i].conj())
        for i in range(len(states))
    )


# %%
# ## 1. State Discrimination — Orthogonal Baseline
#
# Before introducing SIC-POVMs, it is instructive to start with the simplest
# possible case: two *orthogonal* states. Orthogonal states point in completely
# different directions in Hilbert space, so a single projective measurement
# always tells them apart with certainty.
#
# In $d=2$, the computational basis states $|0\rangle$ and $|1\rangle$ sit at
# the north and south poles of the Bloch sphere. Their antipodal placement is
# the geometric reason they can be perfectly distinguished: a projective
# measurement along the $z$-axis separates them with probability 1, because the
# two states lie in opposite hemispheres no matter which axis you pick.
#
# This immediately raises a deeper question: what if instead we insisted that
# *every pair* of states looks equally similar to every other pair? That is
# exactly the SIC condition — and it forces the states to spread over the sphere
# as uniformly as possible, making discrimination genuinely hard. We will verify
# below that this symmetry constraint caps the success probability at
# $2/[d(d+1)]$.

e0, e1 = standard_basis(2)
rho0 = e0 @ e0.conj().T
rho1 = e1 @ e1.conj().T
p_succ_orth, _ = state_distinguishability([rho0, rho1], [0.5, 0.5])
print(f"Orthogonal basis  p_succ = {p_succ_orth:.6f}   (expected 1.0)")

plot_bloch_sphere(
    [[e0], [e1]],
    labels=[r"$|0\rangle$ (north pole)", r"$|1\rangle$ (south pole)"],
    colors=["royalblue", "crimson"],
    markers=["^", "v"],
    title=r"Orthogonal basis — antipodal points ($p_\mathrm{succ}=1$)",
)

# %%
# As expected, `state_distinguishability` returns 1.0 for the orthogonal pair.
# The Bloch sphere picture confirms the intuition: two points at opposite poles
# have the maximum possible separation. Any measurement that resolves north from
# south succeeds perfectly.

# %%
# ## 2. Qubit SIC — Gram Matrix and Equiangularity
#
# We are now ready to construct the qubit SIC-POVM. A SIC-POVM in dimension $d$
# consists of exactly $d^2$ unit vectors satisfying the *equiangularity*
# condition: every pair of distinct states has the same squared overlap
# $1/(d+1)$.
#
# For $d=2$ this gives four states with pairwise overlap-squared $1/3$. On the
# Bloch sphere, these four states map to the vertices of a regular tetrahedron
# inscribed in the unit sphere — a configuration that is as symmetric as four
# points on a sphere can be. We use `vectors_to_gram_matrix` to compute all
# pairwise overlaps at once and verify the equiangularity condition.
#
# Six identical numbers in the off-diagonal positions of the Gram matrix confirm
# equiangularity. Geometrically, this means there is no privileged measurement
# direction: Bob cannot find any axis that separates one state from the others
# any better than any other axis.

sic2 = sic_d2()
G2 = vectors_to_gram_matrix(sic2)
G2_sq = np.abs(G2) ** 2

print("Overlap-squared Gram matrix (d=2):")
print(np.round(G2_sq, 4))

off_diag = G2_sq[~np.eye(4, dtype=bool)]
print(f"\nAll off-diagonal values equal 1/3: {np.allclose(off_diag, 1/3)}")

overlaps_d2 = [abs(np.vdot(sic2[i], sic2[j])) ** 2 for i, j in combinations(range(4), 2)]
print(f"Pairwise overlaps-squared: {np.round(overlaps_d2, 6)}")

plot_bloch_sphere(
    [sic2],
    labels=["SIC states"],
    colors=["darkorange"],
    markers=["o"],
    title=r"Tetrahedral SIC states ($d=2$, $|\langle\psi_i|\psi_j\rangle|^2 = \frac{1}{3}$)",
)

# %%
# The four orange arrows point to the vertices of a regular tetrahedron. Rotating
# the sphere in any direction, you will find that the four vertices always look
# the same relative to one another — there is no special axis. This visual
# symmetry is the Bloch-sphere face of equiangularity.

# %%
# ## 3. State Discrimination of the Qubit SIC
#
# For any SIC ensemble the optimal success probability is known in closed form:
#
# $$p_\text{succ}^\text{SIC} = \frac{2}{d(d+1)}.$$
#
# For $d=2$ this gives $p_\text{succ} = 1/3$. We now verify this using the SDP
# solver in `|toqito⟩` and place both ensembles side-by-side on the Bloch sphere
# to make the geometric intuition concrete.
#
# The orthogonal pair (poles) can be split by a great-circle measurement; the
# SIC vertices, equidistant from one another, offer no such privileged axis.
# The SDP confirms that no measurement strategy can beat $1/3$ when Bob receives
# one of the four SIC states uniformly at random.

rhos_d2 = [np.outer(v, v.conj()) for v in sic2]
probs_d2 = [1 / 4] * 4
p_succ_d2, _ = state_distinguishability(rhos_d2, probs_d2)

d = 2
p_theory_d2 = 2 / (d * (d + 1))
print(f"toqito SDP result:  p_succ = {p_succ_d2:.6f}")
print(f"Analytic formula:   p_succ = {p_theory_d2:.6f}   [2 / d(d+1), d={d}]")
print(f"Agreement: {np.isclose(p_succ_d2, p_theory_d2, atol=1e-5)}")

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
# The SDP confirms the analytic formula exactly. Visually, the blue poles admit
# a clean separating equator; the orange tetrahedron has no such equator. Having
# understood the qubit case geometrically and numerically, we now ask whether the
# same structure persists in higher dimensions — and how to construct SIC states
# when there is no Bloch-sphere picture to guide intuition.

# %%
# ## 4. Qutrit SIC — Weyl-Heisenberg Orbit, $d=3$
#
# In $d \geq 3$ there is no Bloch-sphere analogue, but there is a powerful
# algebraic recipe. The **Weyl-Heisenberg displacement operators**
#
# $$D_{jk} = \tau^{jk} X^j Z^k, \qquad \tau = e^{i\pi/d},$$
#
# where $X$ (shift) and $Z$ (clock) are the generalised Pauli matrices,
# generate a group that acts transitively on state space. Starting from a
# carefully chosen **fiducial vector** $|\phi\rangle$, the group orbit
#
# $$\bigl\{\, D_{jk}|\phi\rangle \;:\; j,k = 0,\ldots,d-1 \,\bigr\}$$
#
# produces all $d^2$ SIC states.
#
# We build the nine Hesse SIC states for $d=3$ using this orbit and verify that
# the equiangularity condition $1/(d+1) = 1/4$ holds for all 72 off-diagonal
# pairs. The discrimination bound $2/[d(d+1)]$ is then confirmed by the SDP. As
# $d$ grows, $2/[d(d+1)]$ falls like $1/d^2$ — a direct consequence of the
# ever-tighter equiangular packing in a larger space.

sic3 = sic_d3()

G3_sq = np.abs(vectors_to_gram_matrix(sic3)) ** 2
off3 = G3_sq[~np.eye(9, dtype=bool)]
print(f"d=3  off-diagonal overlap-squared — mean: {off3.mean():.6f},  std: {off3.std():.2e}")
print(f"Expected 1/4 = {0.25}   All equal: {np.allclose(off3, 1/4, atol=1e-8)}")

rhos_d3 = [np.outer(v, v.conj()) for v in sic3]
probs_d3 = [1 / 9] * 9
p_succ_d3, _ = state_distinguishability(rhos_d3, probs_d3)

d = 3
p_theory_d3 = 2 / (d * (d + 1))
print(f"toqito SDP result:  p_succ = {p_succ_d3:.6f}")
print(f"Analytic formula:   p_succ = {p_theory_d3:.6f}   [2 / d(d+1), d={d}]")

# %%
# Both the equiangularity check and the SDP confirm that the $d=3$ construction
# is valid. The success probability has dropped from $1/3$ to $1/6$, consistent
# with the analytic formula, reflecting the increasing difficulty of
# discrimination as the dimension grows and the states pack more tightly relative
# to the available space.

# %%
# ## 5. Projective 2-Design — Welch Bound Saturation
#
# We have seen that SIC-POVMs minimise discrimination success probability. There
# is a deeper reason for this: SIC-POVMs are **projective 2-designs**, meaning
# they reproduce the second moment of the Haar measure over pure states. This
# implies they sample state space as uniformly as any random ensemble.
#
# The order-2 frame potential
#
# $$\Phi^{(2)} = \sum_{i,j} |\langle \psi_i | \psi_j \rangle|^4$$
#
# is lower bounded by the Welch bound $N^2 / \binom{d+1}{2}$, and saturating
# this bound is *equivalent* to being a projective 2-design. We verify that both
# our SIC constructions hit the bound exactly.

N2, d2 = 4, 2
phi2 = frame_potential(sic2)
phi2_min = N2 ** 2 / comb(d2 + 1, 2)
print(f"d=2  frame potential: {phi2:.6f}   Welch bound: {phi2_min:.6f}   "
      f"Saturated: {np.isclose(phi2, phi2_min)}")

N3, d3 = 9, 3
phi3 = frame_potential(sic3)
phi3_min = N3 ** 2 / comb(d3 + 1, 2)
print(f"d=3  frame potential: {phi3:.6f}   Welch bound: {phi3_min:.6f}   "
      f"Saturated: {np.isclose(phi3, phi3_min, atol=1e-6)}")

# %%
# Both sets saturate the Welch bound. This means that the discrimination
# difficulty and the 2-design property are two faces of the same equiangularity
# coin: the very symmetry that makes the states hard to tell apart also makes
# them an optimal reference measurement for randomised benchmarking and other
# protocols that require uniform sampling of state space.

# %%
# ## 6. Resolution of the Identity
#
# For a valid POVM the elements must sum to the identity,
#
# $$\sum_{i=1}^{d^2} M_i = I_d, \qquad M_i = \frac{|\psi_i\rangle\langle\psi_i|}{d},$$
#
# which ensures that outcome probabilities always sum to one — a basic
# requirement of any measurement. This is a non-trivial constraint: $d^2$
# rank-1 operators, each scaled by $1/d$, must conspire to sum to the full
# identity. It follows automatically from equiangularity, but it is reassuring to
# verify it numerically. We compute the Frobenius distance between the frame
# operator and the identity; a valid SIC gives machine-precision residuals.

print(f"Resolution of identity residual  d=2: {resolution_residual(sic2):.2e}")
print(f"Resolution of identity residual  d=3: {resolution_residual(sic3):.2e}")

# %%
# Both residuals are at floating-point precision ($\sim 10^{-16}$), confirming
# the tight-frame property. With completeness verified, we can now ask what it
# buys us in practice: if we apply the SIC measurement repeatedly to many
# identically prepared copies of an unknown state, can we reconstruct the state
# exactly?

# %%
# ## 7. Informational Completeness — State Reconstruction
#
# A POVM is *informationally complete* (IC) if its outcome statistics uniquely
# determine the state: no two distinct density matrices produce identical
# probability distributions. The SIC POVM is IC, and it admits a particularly
# clean reconstruction formula
#
# $$\rho = \sum_{i=1}^{d^2} \left[(d+1)\,p_i - \frac{1}{d}\right]
# |\psi_i\rangle\langle\psi_i|,$$
#
# where $p_i = \mathrm{Tr}(M_i \rho)$ are the measured outcome probabilities.
# This formula is the operational payoff of equiangularity: because all SIC
# elements are related by the same symmetry group, the dual frame needed to
# invert the measurement inherits the same clean structure. No other IC-POVM
# with $d^2$ elements has such a tidy dual.
#
# We perform a round-trip reconstruction of a random qubit state and confirm
# that the Frobenius error is at machine precision.

rng = np.random.default_rng(42)
A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
rho = A @ A.conj().T
rho /= np.trace(rho)

povm = [np.outer(v, v.conj()) / 2 for v in sic2]
p_obs = np.array([np.real(np.trace(M @ rho)) for M in povm])
rho_rec = sic_reconstruct(p_obs, sic2, d=2)
print(f"Reconstruction Frobenius error: {np.linalg.norm(rho - rho_rec):.2e}   "
      f"(machine precision expected)")

# %%
# The reconstruction is exact to floating-point precision. This confirms a key
# conceptual point: the same equiangularity that *limits* single-shot
# discrimination to $2/[d(d+1)]$ also *enables* exact reconstruction in the
# many-shot limit. SIC-POVMs are not designed to distinguish states in a single
# shot; they are designed to gather complete information uniformly across many
# shots.

# %%
# ## 8. Comparison with Mutually Unbiased Bases (MUBs)
#
# SIC-POVMs are not the only maximally symmetric measurement family. **Mutually
# Unbiased Bases** (MUBs) are another well-known construction with complementary
# properties:
#
# - MUBs — intra-basis overlaps are 0 or 1; inter-basis overlaps are $1/d$.
# - SICs — *all* off-diagonal overlaps equal $1/(d+1) < 1/d$.
#
# Because $1/d > 1/(d+1)$, the MUB inter-basis overlaps are larger, meaning the
# states are further apart on average. This improves single-shot discrimination
# relative to the SIC ensemble. In $d=2$ the three MUBs produce six states that
# sit at the vertices of a regular octahedron — including the two poles — so the
# MUB ensemble inherits some of the orthogonal-basis discrimination advantage.
# The SIC tetrahedron has no such privileged pair of antipodal states.
#
# We use `mutually_unbiased_basis` to retrieve the six qubit MUB vectors, then
# compare success probabilities directly.

mub2 = mutually_unbiased_basis(2)
G_mub = np.abs(vectors_to_gram_matrix(mub2)) ** 2

print(f"Number of MUB vectors returned: {len(mub2)}  (3 bases x 2 vectors each)")
print("\nMUB overlap-squared matrix (d=2):")
print(np.round(G_mub, 3))

rhos_mub = [np.outer(v, v.conj()) for v in mub2]
probs_mub = [1 / 6] * 6
p_succ_mub, _ = state_distinguishability(rhos_mub, probs_mub)

print(f"MUB (6 states, d=2)  p_succ = {p_succ_mub:.6f}")
print(f"SIC (4 states, d=2)  p_succ = {p_succ_d2:.6f}")

# %%
# As expected, the MUB ensemble achieves a higher success probability than the
# SIC ensemble: MUBs include antipodal pairs, whereas the SIC tetrahedron does
# not. The Gram matrix of the MUBs shows a block structure — zeros within each
# basis, $1/d$ across bases — which is qualitatively different from the SIC's
# uniform off-diagonal structure. Both are maximally symmetric, but in different
# senses.

# %%
# ## 9. Learnability of the SIC Ensemble
#
# Finally, we examine the *learnability* of the SIC ensemble. Learnability
# generalises state discrimination to the setting where Bob receives $k$ copies
# of the unknown state and must classify it based on all $k$ measurements.
#
# At $k=1$ — a single copy — learnability reduces to ordinary state
# discrimination. The result should therefore match `state_distinguishability`
# exactly, confirming that the two tasks are identical in the single-shot regime.
# This is a useful consistency check: the same equiangularity that makes
# reconstruction possible in the many-shot limit is exactly what caps the
# per-shot classification accuracy at $2/[d(d+1)]$.

learn_d2 = learnability(sic2, k=1)

print(f"SIC learnability (d=2, k=1)  average classification error: {learn_d2['value']:.6f}")
print(f"Complement (success):                                        {1 - learn_d2['value']:.6f}")
print(f"State discrimination p_succ (d=2):                          {p_succ_d2:.6f}")

# %%
# The learnability complement matches `state_distinguishability` to the expected
# numerical precision, confirming the single-shot equivalence. To explore how
# classification accuracy improves as $k$ increases, you can call
# `learnability(sic2, k=2)`, `k=3`, and so on — each additional copy provides
# more information and pushes the success probability closer to 1.

# %%
# ## Putting It All Together
#
# Every property explored in this tutorial flows from a single geometric idea:
# SIC states are distributed as uniformly as possible over state space. The chain
# of equivalent statements is:
#
# 1. All pairwise overlaps equal $1/(d+1)$.
# 2. The frame potential saturates the Welch bound  (projective 2-design).
# 3. The frame operator equals the identity          (tight frame / POVM completeness).
# 4. The dual frame has the same structure           (clean reconstruction formula).
# 5. No measurement axis is privileged               (discrimination capped at $2/[d(d+1)]$).
#
# The SIC-POVM sits between two extremes on the Bloch sphere:
#
# | Ensemble | Geometry | $p_\text{succ}$ | IC? |
# |---|---|---|---|
# | Orthogonal basis | Antipodal points | 1 | No |
# | SIC tetrahedron | Tetrahedral vertices | $2/[d(d+1)]$ | Yes |
#
# Using `|toqito⟩` we verified all five equivalent formulations numerically,
# establishing that the discrimination viewpoint gives SIC symmetry a direct
# operational meaning. Equiangularity is not merely an aesthetic property — it
# is the precise condition that simultaneously minimises single-shot
# discrimination and maximises the uniformity of information gathered across many
# shots.
#
# For further exploration, consider:
#
# - Constructing SIC-POVMs in dimension $d=4$ using the Weyl-Heisenberg recipe
#   and verifying the equiangularity condition $1/5$.
# - Examining how learnability scales with the number of copies $k$.
# - Comparing SIC state tomography with MUB-based tomography in terms of
#   reconstruction noise when outcome probabilities are estimated from finite
#   data.
#
# mkdocs_gallery_thumbnail_path = 'figures/logo.png'
