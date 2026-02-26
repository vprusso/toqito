"""
============================================================
Quantum classification, factor width, :math:`k`-incoherence
============================================================

This example accompanies the "The complexity of quantum state classification"
paper :footcite:`Johnston_2025_Complexity`.

In this tutorial, we will cover the concepts of the so-called "learnability" of
quantum states along with related settings of "factor width" and the notion of
:math:`k`-incoherence of a matrix. More details can be found in the
aforementioned paper.
"""

# %%
# Learnability of quantum states
# ------------------------------
#
# To illustrate :math:`k`-learnability, consider the following generalization
# of the trine states to four states in three dimensions, called the
# *tetrahedral states*:
#
# .. math::
#   \begin{aligned}
#       \ket{\psi_1} = \frac{1}{\sqrt{3}} (\ket{0} + \ket{1} + \ket{2}), \quad &
#       \ket{\psi_2} = \frac{1}{\sqrt{3}} (\ket{0} - \ket{1} - \ket{2}), \\
#       \ket{\psi_3} = \frac{1}{\sqrt{3}} (-\ket{0} - \ket{1} + \ket{2}), \quad &
#       \ket{\psi_4} = \frac{1}{\sqrt{3}} (-\ket{0} + \ket{1} - \ket{2}).
#   \end{aligned}

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
# This set of states is :math:`2`-learnable, upon receiving one of them, one
# can always guess two states from which it was selected without error.

from toqito.state_props import learnability

states = tetrahedral_states()
learnability_result = learnability(states, k=2)
print(f"Average classification error (k=2): {learnability_result['value']}")

# %%
# Indeed, can be accomplished using the following POVM :math:`M_{i,j} =
# \frac{1}{2} \ket{\phi_{i,j}} \bra{\phi_{i,j}}`, where
#
# .. math::
#   \begin{aligned}
#       \ket{\phi_{1,2}} &= \frac{1}{\sqrt{2}}(\ket{1} + \ket{2}), \quad
#       \ket{\phi_{1,3}} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{1}), \quad
#       \ket{\phi_{1,4}} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{2}), \\
#       \ket{\phi_{2,3}} &= \frac{1}{\sqrt{2}}(\ket{0} - \ket{2}), \quad
#       \ket{\phi_{2,4}} = \frac{1}{\sqrt{2}}(\ket{0} - \ket{1}), \quad
#       \ket{\phi_{3,4}} = \frac{1}{\sqrt{2}}(\ket{1} - \ket{2}).
#   \end{aligned}


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
# By contrast however, these states are not :math:`k=1`-learnable:

states = tetrahedral_states()
learnability_result = learnability(states, k=1)
print(f"Average classification error (k=1): {learnability_result['value']}")

# %%
# :math:`k`-Incoherence
# ----------------------
# The notion of :math:`k`-incoherence comes from
# :footcite:`Johnston_2022_Absolutely`. For a positive integers, :math:`k` and
# :math:`n`, the matrix :math:`X \in \text{Pos}(\mathbb{C}^n)` is called
# :math:`k`-incoherent if there exists a positive integer :math:`m`, a set
# :math:`S = \{|\psi_0\rangle, |\psi_1\rangle,\ldots, |\psi_{m-1}\rangle\}
# \subset \mathbb{C}^n` with the property that each :math:`|\psi_i\rangle` has
# at most :math:`k` non-zero entries, and real scalars :math:`c_0, c_1, \ldots,
# c_{m-1} \geq 0` for which
#
# .. math::
#     X = \sum_{j=0}^{m-1} c_j |\psi_j\rangle \langle \psi_j|.
#
# This function checks if the provided density matrix :code:`mat` is
# k-incoherent. It returns True if :code:`mat` is k-incoherent and False if
# :code:`mat` is not.
#
# For example, the following matrix is :math:`2`-incoherent
#
# .. math::
#       \begin{pmatrix}
#           2 & 1 & 2 \\ 1 & 2 & -1 \\ 2 & -1 & 5
#       \end{pmatrix}
#
# Indeed, one can verify this numerically using the
# :py:func:`~toqito.matrix_props.is_k_incoherent.is_k_incoherent`.

from toqito.matrix_props import is_k_incoherent


mat = np.array([[2, 1, 2], [1, 2, -1], [2, -1, 5]])
print(is_k_incoherent(mat, 2))

# %%
# Factor width
# -------------
#
# Another closely related definition to :math:`k`-incoherence is that of
# factor width :footcite:`Barioli_2003_Maximal, Johnston_2025_Factor,
# Boman_2005_factor` below.
#
# Let :math:`k` be a positive integer. The factor width of a positive
# semidefinite matrix :math:`X` is the smallest :math:`k` such that it is
# :math:`k`-incoherent.
#
# For example, the matrix :math:`\operatorname{diag}(1, 1, 0)` has factor width
# at most :math:`1`.

from toqito.matrix_props import factor_width

diag_mat = np.diag([1, 1, 0])
result = factor_width(diag_mat, k=1)
print(result["feasible"])


# %%
# Conversely, the rank-one matrix :math:`\frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1
# & 1 \end{pmatrix}` is not :math:`1`-factorable.

hadamard = np.array([[1, 1], [1, 1]], dtype=np.complex128) / 2
result = factor_width(hadamard, k=1)
print(result["feasible"])

# %%
# This example comes directly from :footcite:`Johnston_2025_Factor`. Suppose we want to determine the factor width of
# the rank-:math:`3` matrix
#
# .. math::
#       M = \begin{bmatrix}
#        2 & 1 & 1 & -1 \\
#        1 & 2 & 0 & 1 \\
#        1 & 0 & 2 & -1 \\
#        -1 & 1 & -1 & 2
#       \end{bmatrix}.
#
# We start by finding a basis for :math:`S := \text{range}(M)`, which can be done by picking a linearly independent set
# of :math:`r = 3` columns of :math:`M`: :math:`S = \operatorname{span}\{(2,1,1,-1), (1,2,0,1), (1,0,2,-1)\}`. Then
# :math:`R_0 = \{S\}` and we proceed recursively:
#
# .. math::
#   \begin{aligned}
#       R_1 = \{S_1, S_2, S_3, S_3\}, \quad \text{where} \quad S_1 & = \operatorname{span}\{(0,1,-1,1), (0,1,-3,1)\}, \\
#       S_2 & = \operatorname{span}\{(1,0,2,-1), (3,0,2,-3)\}, \\
#       S_3 & = \operatorname{span}\{(1,2,0,1), (3,2,0,-1)\}, \ \ \text{and} \\
#       S_4 & = \operatorname{span}\{(1,1,1,0), (3,3,1,0)\}.
#   \end{aligned}
#
# To determine whether or not :math:`M` is :math:`3`-incoherent, we let :math:`\Pi_1`, :math:`\Pi_2`, :math:`\Pi_3`, and
# :math:`\Pi_4` be the orthogonal projections onto :math:`S_1`, :math:`S_2`, :math:`S_3`, and :math:`S_4`, respectively.
# We then use semidefinite programming to determine whether or not there exist matrices :math:`M_1, M_2, M_3, M_4 \in
# \text{Pos}(\mathbb{C}^4)` for which
#
# .. math::
#       M = M_1 + M_2 + M_3 + M_4, \quad \text{and} \quad M_j = \Pi_j M_j \Pi_j \quad \text{for all} \quad j \in \{1,2,3,4\}.
#
# Indeed, such matrices do exist:
#
# .. math::
#        M_1 = \begin{bmatrix}
#            0 & 0 & 0 & 0 \\
#            0 & 1 & -1 & 1 \\
#            0 & -1 & 1 & -1 \\
#            0 & 1 & -1 & 1
#        \end{bmatrix}, \ M_2 = \begin{bmatrix}
#            1 & 0 & 0 & -1 \\
#            0 & 0 & 0 & 0 \\
#            0 & 0 & 0 & 0 \\
#            -1 & 0 & 0 & 1
#        \end{bmatrix}, \ M_3 = \begin{bmatrix}
#            0 & 0 & 0 & 0 \\
#            0 & 0 & 0 & 0 \\
#            0 & 0 & 0 & 0 \\
#            0 & 0 & 0 & 0
#        \end{bmatrix}, \ M_4 = \begin{bmatrix}
#            1 & 1 & 1 & 0 \\
#            1 & 1 & 1 & 0 \\
#            1 & 1 & 1 & 0 \\
#            0 & 0 & 0 & 0
#        \end{bmatrix},
#
# so :math:`M` is :math:`3`-incoherent. For example, we can verify this numerically using the
# :py:func:`~toqito.matrix_props.factor_width.factor_width`.

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
# To similarly determine whether or not :math:`M` is :math:`2`-incoherent, we proceed further with the recursive
# construction by computing
#
# .. math::
#
#    R_2 = \{S_{\{1,2\}}, S_{\{1,3\}}, S_{\{2,3\}}, S_{\{3,4\}}\}, \quad \text{where} \quad S_{\{1,2\}} = S_{\{1,4\}} = S_{\{2,4\}} = \operatorname{span}\{(0,0,1,0)\},
#
#    S_{\{1,3\}} = \operatorname{span}\{(0,1,0,1)\}, \quad S_{\{2,3\}} = \operatorname{span}\{(1,0,0,-1)\}, \quad \text{and} \quad S_{\{3,4\}} = \operatorname{span}\{(1,1,0,0)\}.
#
# It follows that the only vectors in :math:`\text{range}(M)` with :math:`k = 2` or fewer non-zero entries are the scalar
# multiples of :math:`{v_1} := (0,0,1,0)`, :math:`{v_2} := (0,1,0,1)`, :math:`{v_3} := (1,0,0,-1)`, and :math:`{v_4} :=
# (1,1,0,0)`, so :math:`M` is :math:`2`-incoherent if and only if there exist non-negative real scalars :math:`c_1`,
# :math:`c_2`, :math:`c_3`, and :math:`c_4` for which
#
# .. math::
#
#    M = c_1 v_1 v_1^* + c_2 v_2 v_2^* + c_3 v_3 v_3^* + c_4 v_4 v_4^*.
#
# It is straightforward to use semidefinite programming (or even just solve by hand in this small example) to see
# that no such scalars exist, so :math:`X` is not :math:`2`-incoherent. It follows that :math:`X` has factor width
# :math:`3`.

result = factor_width(mat, k=2)
print(result["feasible"])
