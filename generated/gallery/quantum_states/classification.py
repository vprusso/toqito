"""# Quantum classification, factor width, k-incoherence

Explores k-learnability of quantum states, factor width of positive matrices,
and k-incoherence. Accompanies the paper on the complexity of quantum state
classification and uses toqito to verify theoretical bounds via semidefinite
programming.
"""

# %%
# ## Learnability of quantum states
#
# To illustrate $k$-learnability, consider the following generalization
# of the trine states to four states in three dimensions, called the
# *tetrahedral states*:
#
# $$
# \begin{aligned}
# \ket{\psi_1} = \frac{1}{\sqrt{3}} (\ket{0} + \ket{1} + \ket{2}), \quad &
# \ket{\psi_2} = \frac{1}{\sqrt{3}} (\ket{0} - \ket{1} - \ket{2}), \\
# \ket{\psi_3} = \frac{1}{\sqrt{3}} (-\ket{0} - \ket{1} + \ket{2}), \quad &
# \ket{\psi_4} = \frac{1}{\sqrt{3}} (-\ket{0} + \ket{1} - \ket{2}).
# \end{aligned}
# $$
#
import numpy as np


def tetrahedral_states() -> list[np.ndarray]:
    return [
        np.array([1, 1, 1], dtype=np.complex128) / np.sqrt(3),
        np.array([1, -1, -1], dtype=np.complex128) / np.sqrt(3),
        np.array([-1, -1, 1], dtype=np.complex128) / np.sqrt(3),
        np.array([-1, 1, -1], dtype=np.complex128) / np.sqrt(3),
    ]


print(tetrahedral_states())

# %%
# This set of states is $2$-learnable, upon receiving one of them, one
# can always guess two states from which it was selected without error.

from toqito.state_props import learnability

states = tetrahedral_states()
learnability_result = learnability(states, k=2)
print(f"Average classification error (k=2): {learnability_result['value']}")


# %%
# Indeed, can be accomplished using the following POVM $M_{i,j} =
# \frac{1}{2} \ket{\phi_{i,j}} \bra{\phi_{i,j}}$, where
#
# $$
# \begin{aligned}
# \ket{\phi_{1,2}} &= \frac{1}{\sqrt{2}}(\ket{1} + \ket{2}), \quad
# \ket{\phi_{1,3}} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{1}), \quad
# \ket{\phi_{1,4}} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{2}), \\
# \ket{\phi_{2,3}} &= \frac{1}{\sqrt{2}}(\ket{0} - \ket{2}), \quad
# \ket{\phi_{2,4}} &= \frac{1}{\sqrt{2}}(\ket{0} - \ket{1}), \quad
# \ket{\phi_{3,4}} = \frac{1}{\sqrt{2}}(\ket{1} - \ket{2}).
# \end{aligned}
# $$
#
def povm_residual(states: list[np.ndarray], povm: dict[tuple[int, int], np.ndarray]) -> tuple[float, float]:
    """Return the maximum POVM reconstruction and support violations."""
    dim = states[0].shape[0]
    total = sum(povm.values(), np.zeros((dim, dim), dtype=np.complex128))
    sum_residual = np.max(np.abs(total - np.eye(dim)))

    zero_residual = 0.0
    for idx, state in enumerate(states):
        for subset, operator in povm.items():
            if idx not in subset:
                zero_residual = max(zero_residual, np.abs(np.vdot(state, operator @ state)))
    return sum_residual, zero_residual


phi_vectors = {
    (0, 1): np.array([0, 1, 1], dtype=np.complex128) / np.sqrt(2),
    (0, 2): np.array([1, 1, 0], dtype=np.complex128) / np.sqrt(2),
    (0, 3): np.array([1, 0, 1], dtype=np.complex128) / np.sqrt(2),
    (1, 2): np.array([1, 0, -1], dtype=np.complex128) / np.sqrt(2),
    (1, 3): np.array([1, -1, 0], dtype=np.complex128) / np.sqrt(2),
    (2, 3): np.array([0, 1, -1], dtype=np.complex128) / np.sqrt(2),
}
povm_elements = {pair: 0.5 * np.outer(vec, vec.conj()) for pair, vec in phi_vectors.items()}

sum_res, zero_res = povm_residual(states, povm_elements)
print(f"max|Σ M_S - I|             : {sum_res:.2e}")
print(f"max|⟨ψ_i|M_S|ψ_i⟩| (i∉S)   : {zero_res:.2e}")

# %%
# By contrast however, these states are not $k=1$-learnable:

states = tetrahedral_states()
learnability_result = learnability(states, k=1)
print(f"Average classification error (k=1): {learnability_result['value']}")

# %%
# ## k-Incoherence
# The notion of $k$-incoherence comes from
# [@johnston2022absolutely]. For a positive integers, $k$ and
# $n$, the matrix $X \in \text{Pos}(\mathbb{C}^n)$ is called
# $k$-incoherent if there exists a positive integer $m$, a set
# $S = \{|\psi_0\rangle, |\psi_1\rangle,\ldots, |\psi_{m-1}\rangle\}
# \subset \mathbb{C}^n$ with the property that each $|\psi_i\rangle$ has
# at most $k$ non-zero entries, and real scalars $c_0, c_1, \ldots,
# c_{m-1} \geq 0$ for which
#
# $$
# X = \sum_{j=0}^{m-1} c_j |\psi_j\rangle \langle \psi_j|.
# $$
#
# This function checks if the provided density matrix `mat` is
# k-incoherent. It returns True if `mat` is k-incoherent and False if
# `mat` is not.
#
# For example, the following matrix is $2$-incoherent
#
# $$
# \begin{pmatrix}
# 2 & 1 & 2 \\ 1 & 2 & -1 \\ 2 & -1 & 5
# \end{pmatrix}
# $$
#
# Indeed, one can verify this numerically using the [`is_k_incoherent`][toqito.matrix_props.is_k_incoherent.is_k_incoherent].
#
from toqito.matrix_props import is_k_incoherent

mat = np.array([[2, 1, 2], [1, 2, -1], [2, -1, 5]])
print(is_k_incoherent(mat, 2))

# %%
# ## Factor width
#
# Another closely related definition to $k$-incoherence is that of
# factor width [@barioli2003maximal][@johnston2025factor][@boman2005factor] below.
#
# Let $k$ be a positive integer. The factor width of a positive
# semidefinite matrix $X$ is the smallest $k$ such that it is
# $k$-incoherent.
#
# For example, the matrix $\operatorname{diag}(1, 1, 0)$ has factor width
# at most $1$.

from toqito.matrix_props import factor_width

diag_mat = np.diag([1, 1, 0])
result = factor_width(diag_mat, k=1)
print(result["feasible"])


# %%
# Conversely, the rank-one matrix $\frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1
# & 1 \end{pmatrix}$ is not $1$-factorable.

hadamard = np.array([[1, 1], [1, 1]], dtype=np.complex128) / 2
result = factor_width(hadamard, k=1)
print(result["feasible"])

# %%
# This example comes directly from [@johnston2025factor]. Suppose we want to determine the factor width of
# the rank-$3$ matrix
#
# $$
# M = \begin{bmatrix}
# 2 & 1 & 1 & -1 \\
# 1 & 2 & 0 & 1 \\
# 1 & 0 & 2 & -1 \\
# -1 & 1 & -1 & 2
# \end{bmatrix}.
# $$
#
# We start by finding a basis for $S := \text{range}(M)$, which can be done by picking a linearly independent set
# of $r = 3$ columns of $M$: $S = \operatorname{span}\{(2,1,1,-1), (1,2,0,1), (1,0,2,-1)\}$. Then
# $R_0 = \{S\}$ and we proceed recursively:
#
# $$
# \begin{aligned}
# R_1 = \{S_1, S_2, S_3, S_3\}, \quad \text{where} \quad S_1 & = \operatorname{span}\{(0,1,-1,1), (0,1,-3,1)\}, \\
# S_2 & = \operatorname{span}\{(1,0,2,-1), (3,0,2,-3)\}, \\
# S_3 & = \operatorname{span}\{(1,2,0,1), (3,2,0,-1)\}, \ \ \text{and} \\
# S_4 & = \operatorname{span}\{(1,1,1,0), (3,3,1,0)\}.
# \end{aligned}
# $$
#
# To determine whether or not $M$ is $3$-incoherent, we let $\Pi_1$, $\Pi_2$, $\Pi_3$, and
# $\Pi_4$ be the orthogonal projections onto $S_1$, $S_2$, $S_3$, and $S_4$, respectively.
# We then use semidefinite programming to determine whether or not there exist matrices $M_1, M_2, M_3, M_4 \in
# \text{Pos}(\mathbb{C}^4)$ for which
#
# $$
# M = M_1 + M_2 + M_3 + M_4, \quad \text{and} \quad M_j = \Pi_j M_j \Pi_j \quad \text{for all} \quad j \in \{1,2,3,4\}.
# $$
#
# Indeed, such matrices do exist:
#
# $$
# M_1 = \begin{bmatrix}
# 0 & 0 & 0 & 0 \\
# 0 & 1 & -1 & 1 \\
# 0 & -1 & 1 & -1 \\
# 0 & 1 & -1 & 1
# \end{bmatrix}, \ M_2 = \begin{bmatrix}
# 1 & 0 & 0 & -1 \\
# 0 & 0 & 0 & 0 \\
# 0 & 0 & 0 & 0 \\
# -1 & 0 & 0 & 1
# \end{bmatrix}, \ M_3 = \begin{bmatrix}
# 0 & 0 & 0 & 0 \\
# 0 & 0 & 0 & 0 \\
# 0 & 0 & 0 & 0 \\
# 0 & 0 & 0 & 0
# \end{bmatrix}, \ M_4 = \begin{bmatrix}
# 1 & 1 & 1 & 0 \\
# 1 & 1 & 1 & 0 \\
# 1 & 1 & 1 & 0 \\
# 0 & 0 & 0 & 0
# \end{bmatrix},
# $$
#
# so $M$ is $3$-incoherent. For example, we can verify this numerically using the [`factor_width`][toqito.matrix_props.factor_width.factor_width].
#
mat = np.array(
    [
        [2, 1, 1, -1],
        [1, 2, 0, 1],
        [1, 0, 2, -1],
        [-1, 1, -1, 2],
    ],
    dtype=np.complex128,
)
result = factor_width(mat, k=3)
print(sum(result["factors"]))

# %%
# To similarly determine whether or not $M$ is $2$-incoherent, we proceed further with the recursive
# construction by computing
#
# $$
# R_2 = \{S_{\{1,2\}}, S_{\{1,3\}}, S_{\{2,3\}}, S_{\{3,4\}}\}, \quad \text{where} \quad S_{\{1,2\}} = S_{\{1,4\}} = S_{\{2,4\}} = \operatorname{span}\{(0,0,1,0)\},
# $$
#
# $$
# S_{\{1,3\}} = \operatorname{span}\{(0,1,0,1)\}, \quad S_{\{2,3\}} = \operatorname{span}\{(1,0,0,-1)\}, \quad \text{and} \quad S_{\{3,4\}} = \operatorname{span}\{(1,1,0,0)\}.
# $$
#
# It follows that the only vectors in $\text{range}(M)$ with $k = 2$ or fewer non-zero entries are the scalar
# multiples of ${v_1} := (0,0,1,0)$, ${v_2} := (0,1,0,1)$, ${v_3} := (1,0,0,-1)$, and ${v_4} :=
# (1,1,0,0)$, so $M$ is $2$-incoherent if and only if there exist non-negative real scalars $c_1$,
# $c_2$, $c_3$, and $c_4$ for which
#
# $$
# M = c_1 v_1 v_1^* + c_2 v_2 v_2^* + c_3 v_3 v_3^* + c_4 v_4 v_4^*.
# $$
#
# It is straightforward to use semidefinite programming (or even just solve by hand in this small example) to see
# that no such scalars exist, so $X$ is not $2$-incoherent. It follows that $X$ has factor width
# $3$.
#
result = factor_width(mat, k=2)
print(result["feasible"])
# mkdocs_gallery_thumbnail_path = 'figures/classification_tetrahedron.png'
