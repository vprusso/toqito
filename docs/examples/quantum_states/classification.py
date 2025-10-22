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

# Use the saved Bloch sphere render as the gallery thumbnail.
# sphinx_gallery_thumbnail_path = "figures/classification_tetrahedron.png"

# %%
# Learnability of quantum states
# ------------------------------
#
# To illustrate :math:`k`-learnability, consider the following generalization
# of the trine states to four states in three dimensions, called the
# *tetrahedral states*:
#
# .. math::
#   \begin{equation}
#       \begin{aligned}
#           \ket{\psi_1} = \frac{1}{\sqrt 3} (\ket{0} + \ket{1} + \ket{2}), \quad & 
#           \ket{\psi_2} = \frac{1}{\sqrt 3} (\ket{0} - \ket{1} - \ket{2}), \\ 
#           \ket{\psi_3} = \frac{1}{\sqrt 3} (-\ket{0} - \ket{1} + \ket{2}), \quad & 
#           \ket{\psi_4} = \frac{1}{\sqrt 3} (-\ket{0} + \ket{1} - \ket{2}). 
#       \end{aligned}
#   \end{equation}

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
#   \begin{equation}
#       \begin{aligned}
#           \ket{\phi_{1,2}} &= \frac{1}{\sqrt{2}}(\ket{1} + \ket{2}), \quad
#           \ket{\phi_{1,3}} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{1}), \quad
#           \ket{\phi_{1,4}} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{2}), \\
#           \ket{\phi_{2,3}} &= \frac{1}{\sqrt{2}}(\ket{0} - \ket{2}), \quad
#           \ket{\phi_{2,4}} = \frac{1}{\sqrt{2}}(\ket{0} - \ket{1}), \quad
#           \ket{\phi_{3,4}} = \frac{1}{\sqrt{2}}(\ket{1} - \ket{2}).
#       \end{aligned}
#   \end{equation}

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
# footcite:`Johnston_2022_Absolutely`. For a positive integers, :math:`k` and
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
#   \begin{equation}
#       \begin{pmatrix}
#           2 & 1 & 2 \\ 1 & 2 & -1 \\ 2 & -1 & 5
#       \end{pmatrix}
#   \end{equation}
# 
# Indeed, one can verify this numerically using the
# :func:`toqito.matrix_props.is_k_incoherent` function.

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
