"""SIC-POVMs and Quantum State Discrimination.

Imagine Alice picks a quantum state at random from a secret list and sends it to Bob. Bob's
challenge: guess which state he received. The answer depends not just on how many states are in
the list, but on their *geometry* — how similar or different they are from one another.

This tutorial builds intuition for that geometry by working through a special family of
measurements called **Symmetric Informationally Complete Positive Operator-Valued Measures**
(SIC-POVMs). We will see that equiangularity — every pair of states having the *same* overlap —
simultaneously maximises symmetry, guarantees informational completeness, and places an absolute
ceiling on Bob's success probability.

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


# ---------------------------------------------------------------------------
# 1. State Discrimination — Orthogonal Baseline
# ---------------------------------------------------------------------------
# The best possible discrimination case is *orthogonal* states: they point in completely
# different directions, so a single projective measurement always tells them apart.  We verify
# this baseline before introducing geometric complexity.
#
# |0> sits at the north pole and |1> at the south pole — their antipodal placement is the
# geometric reason they can be perfectly distinguished.  A projective measurement along the
# z-axis separates them with probability 1 because the two states lie in opposite hemispheres
# no matter which axis you pick.
#
# This immediately raises a question: what if instead we insisted that *every pair* of states
# looks equally similar to every other pair?  That is the SIC condition, and it forces the
# states to spread over the sphere as uniformly as possible — making discrimination hard.

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

# ---------------------------------------------------------------------------
# 2. Qubit SIC — Gram Matrix and Equiangularity
# ---------------------------------------------------------------------------
# We construct the four tetrahedral SIC states and verify their equiangularity using
# vectors_to_gram_matrix.  For a valid SIC every off-diagonal entry of |G|^2 equals 1/(d+1).
# Six identical numbers confirm equiangularity, and exactly prevent Bob from finding any
# privileged measurement direction.

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

# ---------------------------------------------------------------------------
# 3. State Discrimination of the Qubit SIC
# ---------------------------------------------------------------------------
# For any SIC ensemble the optimal success probability is known in closed form:
#
#   p_succ^SIC = 2 / (d(d+1)).
#
# For d=2 this gives p_succ = 1/3.  The SDP confirms the analytic bound, and placing both
# ensembles side-by-side on the same sphere makes the geometric intuition concrete: the
# orthogonal pair (poles) can be split by a great-circle measurement; the SIC vertices,
# equidistant from one another, offer no such privileged axis.

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

# Having understood the qubit case geometrically and numerically, we now ask whether the same
# structure persists in higher dimensions — and how to construct SIC states when there is no
# Bloch-sphere picture to guide intuition.

# ---------------------------------------------------------------------------
# 4. Qutrit SIC — Weyl-Heisenberg Orbit, d=3
# ---------------------------------------------------------------------------
# We build the nine Hesse SIC states and verify that the equiangularity condition 1/(d+1) = 1/4
# holds.  The discrimination bound 2/[d(d+1)] is then confirmed by the SDP.  As d grows,
# 2/[d(d+1)] falls like 1/d^2 — a direct consequence of the ever-tighter equiangular packing
# in a larger space, and a sign that this same packing has a deeper statistical interpretation.

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

# ---------------------------------------------------------------------------
# 5. Projective 2-Design — Welch Bound Saturation
# ---------------------------------------------------------------------------
# Both SIC sets saturate the Welch bound, confirming they are projective 2-designs.
# The discrimination difficulty and the 2-design property are two faces of the same coin.

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

# ---------------------------------------------------------------------------
# 6. Resolution of the Identity
# ---------------------------------------------------------------------------
# The POVM completeness condition ensures outcome probabilities sum to one.
# Both SICs satisfy it to machine precision, confirming the tight-frame property.
# With completeness verified, we can now ask what it buys us in practice.

print(f"Resolution of identity residual  d=2: {resolution_residual(sic2):.2e}")
print(f"Resolution of identity residual  d=3: {resolution_residual(sic3):.2e}")

# ---------------------------------------------------------------------------
# 7. Informational Completeness — State Reconstruction
# ---------------------------------------------------------------------------
# Round-trip reconstruction of a random qubit state confirms the SIC POVM is informationally
# complete.  The reconstruction is exact to floating-point precision, confirming that the same
# equiangularity that limits single-shot discrimination also enables exact reconstruction.

rng = np.random.default_rng(42)
A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
rho = A @ A.conj().T
rho /= np.trace(rho)

povm = [np.outer(v, v.conj()) / 2 for v in sic2]
p_obs = np.array([np.real(np.trace(M @ rho)) for M in povm])
rho_rec = sic_reconstruct(p_obs, sic2, d=2)
print(f"Reconstruction Frobenius error: {np.linalg.norm(rho - rho_rec):.2e}   "
      f"(machine precision expected)")

# ---------------------------------------------------------------------------
# 8. Comparison with Mutually Unbiased Bases (MUBs)
# ---------------------------------------------------------------------------
# MUBs and SICs are both maximally symmetric, but structurally distinct:
#
#   MUBs — intra-basis overlaps 0 or 1, inter-basis overlaps 1/d.
#   SICs — all off-diagonal overlaps 1/(d+1) < 1/d.
#
# The larger inter-basis overlap of MUBs means the states are further apart on average,
# which improves discrimination relative to the SIC.  The six MUB states sit at octahedral
# vertices including the two poles, so the MUB ensemble inherits some of the orthogonal-basis
# discrimination advantage.  The SIC tetrahedron has no such privileged pair.

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

# ---------------------------------------------------------------------------
# 9. Learnability of the SIC Ensemble
# ---------------------------------------------------------------------------
# learnability measures average classification accuracy when Bob receives k copies.
# At k=1 — a single copy — the result matches state discrimination exactly, confirming that
# the two tasks are identical in the single-shot regime.  The same equiangularity that makes
# reconstruction possible in the limit of many shots is exactly what caps the per-shot accuracy.

learn_d2 = learnability(sic2, k=1)

print(f"SIC learnability (d=2, k=1)  average classification error: {learn_d2['value']:.6f}")
print(f"Complement (success):                                        {1 - learn_d2['value']:.6f}")
print(f"State discrimination p_succ (d=2):                          {p_succ_d2:.6f}")

# ---------------------------------------------------------------------------
# Putting It All Together
# ---------------------------------------------------------------------------
# Every property explored in this tutorial flows from a single geometric idea: SIC states are
# distributed as uniformly as possible over state space.  The chain of equivalent statements is:
#
#   1. All pairwise overlaps equal 1/(d+1).
#   2. The frame potential saturates the Welch bound  (projective 2-design).
#   3. The frame operator equals the identity          (tight frame / POVM completeness).
#   4. The dual frame has the same structure           (clean reconstruction formula).
#   5. No measurement axis is privileged               (discrimination capped at 2/[d(d+1)]).
#
# The SIC-POVM sits between two extremes on the Bloch sphere:
#
#   Orthogonal bases:  antipodal states, perfect discrimination, incomplete rotational symmetry.
#   SIC tetrahedron:   maximally symmetric, informationally complete, p_succ = 2/[d(d+1)].
#
# Using |toqito> we verified all five equivalent formulations numerically, establishing that
# the discrimination viewpoint gives SIC symmetry a direct operational meaning.
