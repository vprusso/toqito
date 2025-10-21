"""
=================================================
Quantum classification, factor width, incoherence
=================================================

This example accompanies the "The complexity of quantum state classification"
paper :footcite:``

In this tutorial, we will explore the Pusey-Barrett-Rudolph (PBR) theorem
:footcite:`Pusey_2012_On`, a significant no-go theorem in the foundations
of quantum mechanics. We will describe the theorem's core argument and then
use :code:`|toqito⟩` to verify the central mathematical property that
the theorem relies on.

The PBR theorem addresses a fundamental question: Is the quantum state
(e.g., the wavefunction :math:`|\\psi\\rangle`) a real, objective property of a
single system (an *ontic* state), or does it merely represent our
incomplete knowledge or information about some deeper underlying reality
(an *epistemic* state)?
"""

# %%
# Learnability of quantum states
# ------------------------------
#
# To illustrate :math:`k`-learnability, cosnider the following generalization
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
# Factor width of the Gram matrix
# --------------------------------
# The same tetrahedral ensemble has Gram matrix
#
# .. math::
#    G_{ij} = \langle \psi_i, \psi_j \rangle,
#
# and the paper shows this matrix has factor width 2. We confirm this using
# :func:`toqito.matrix_props.factor_width`.

from toqito.matrix_props import factor_width


gram = np.array(
    [[np.vdot(psi_i, psi_j) for psi_j in states] for psi_i in states], 
    dtype=np.complex128
)
factor_width_result = factor_width(gram, k=2)

print(f"Factor-width decomposition feasible? {factor_width_result['feasible']}")
if factor_width_result["feasible"] and factor_width_result["factors"]:
    reconstructed = sum((mat for mat in factor_width_result["factors"] if mat is not None), np.zeros_like(gram))
    recon_residual = np.max(np.abs(reconstructed - gram))
    print(f"Reconstruction residual max|.|     : {recon_residual:.2e}")

# %%
# :math:`k`-Incoherence diagnostics
# ----------------------------------
# The manuscript links factor width to $k$-incoherence.  We compare three
# matrices: the Gram matrix above, a diagonal PSD matrix, and a dense PSD
# matrix that violates the criterion for $k=2$.

from toqito.matrix_props import is_k_incoherent


diagonal_psd = np.diag([0.5, 0.3, 0.2])
dense_psd = np.array([[0.8, 0.4, 0.4], [0.4, 0.3, 0.2], [0.4, 0.2, 0.3]], dtype=np.float64)

print(f"Tetrahedral Gram matrix 2-incoherent? {is_k_incoherent(gram, 2)}")
print(f"Diagonal PSD matrix      1-incoherent? {is_k_incoherent(diagonal_psd, 1)}")
print(f"Dense PSD matrix         2-incoherent? {is_k_incoherent(dense_psd, 2)}")

