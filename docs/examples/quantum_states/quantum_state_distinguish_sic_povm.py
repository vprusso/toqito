"""
Quantum state distinguishability with a SIC-POVM ensemble
=========================================================

In this tutorial we study the problem of *quantum state distinguishability*
using a highly symmetric and geometrically motivated ensemble of quantum
states: a **symmetric informationally complete POVM (SIC-POVM)**.

SIC-POVMs play a central role in quantum information theory and provide
canonical examples of non-orthogonal but maximally symmetric sets of states.
They are discussed extensively in the book *Geometry of Quantum States*
by Bengtsson and Życzkowski.

Here, we focus on the qubit (dimension 2) SIC-POVM, whose four pure states
correspond to the vertices of a regular tetrahedron on the Bloch sphere.
We use this ensemble to demonstrate how :code:`|toqito⟩` can be used to
compute optimal state distinguishability via semidefinite programming.
"""
# %%
# Further background on quantum state distinguishability can be found in
# :footcite:`Watrous_2018_TQI` and the lecture notes :footcite:`Sikora_2019_Semidefinite`.
#
# The state distinguishability problem
# -----------------------------------
#
# Alice possesses an ensemble of quantum states
#
# .. math::
#
#     \eta = \{ (p_0, \rho_0), \ldots, (p_n, \rho_n) \},
#
# where state :math:`\rho_i` is prepared with probability :math:`p_i`.
# Alice sends one state from the ensemble to Bob, who does not know which
# state was sent.
#
# Bob performs a measurement in order to guess which state he received.
# The goal is to maximize the probability of a correct guess.
#
# In the most general setting, the optimal success probability is given
# by the solution to a semidefinite program (SDP) and can be computed
# using :code:`|toqito⟩`.
#
# In this tutorial, we apply this framework to a **geometrically structured**
# ensemble given by a SIC-POVM.

import numpy as np
from toqito.state_opt import state_distinguishability

# %%
# Constructing the qubit SIC-POVM
# -------------------------------
#
# For a SIC-POVM in dimension :math:`d`, the defining property is that the
# corresponding pure states satisfy
#
# .. math::
#
#     \operatorname{Tr}(\rho_i \rho_j) = \frac{1}{d+1}, \quad i \neq j.
#
# In the qubit case (:math:`d = 2`), this gives
#
# .. math::
#
#     \operatorname{Tr}(\rho_i \rho_j) = \frac{1}{3}.
#
# The qubit SIC-POVM consists of four pure states whose Bloch vectors form
# a regular tetrahedron.


def qubit_sic_states():
    """Return the four qubit SIC-POVM density matrices."""

    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    identity = np.eye(2)

    # Tetrahedral Bloch vectors
    bloch_vectors = [
        np.array([1, 1, 1]),
        np.array([1, -1, -1]),
        np.array([-1, 1, -1]),
        np.array([-1, -1, 1]),
    ]

    states = []
    for vec in bloch_vectors:
        vec = vec / np.linalg.norm(vec)
        rho = 0.5 * (
            identity
            + vec[0] * sigma_x
            + vec[1] * sigma_y
            + vec[2] * sigma_z
        )
        states.append(rho)

    return states


states = qubit_sic_states()

# %%
# Verifying the SIC overlap condition
# ----------------------------------
#
# We verify that the SIC-POVM satisfies the defining overlap relation
# :math:`\operatorname{Tr}(\rho_i \rho_j) = 1/3` for :math:`i \neq j`.

overlaps = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        overlaps[i, j] = np.real(np.trace(states[i] @ states[j]))

print("SIC overlap matrix Tr(ρ_i ρ_j):")
print(np.round(overlaps, 4))

# %%
# Optimal global state distinguishability
# ---------------------------------------
#
# We now compute the optimal probability with which Bob can distinguish
# the four SIC states when they are selected with equal probability.
#
# This corresponds to optimizing over all possible quantum measurements.

probs = [1 / 4] * 4

opt_val, _ = state_distinguishability(states, probs)

print(f"Optimal global distinguishability: {opt_val:.4f}")

# %%
# Binary (Helstrom) comparison
# ----------------------------
#
# As a comparison, we consider distinguishing only two of the SIC states
# with equal prior probabilities. In this case, the optimal success
# probability is given by the Helstrom bound.

binary_states = [states[0], states[1]]
binary_probs = [1 / 2, 1 / 2]

binary_val, _ = state_distinguishability(binary_states, binary_probs)

print(f"Binary distinguishability (Helstrom): {binary_val:.4f}")

# %%
# Discussion
# ----------
#
# This example illustrates how geometric symmetry constrains operational
# distinguishability:
#
# * The four SIC states are maximally symmetric but non-orthogonal.
# * Their global distinguishability is strictly less than 1.
# * In the binary case, the distinguishability matches the Helstrom bound.
#
# SIC-POVMs therefore provide a natural and well-structured test case
# for studying quantum state discrimination using semidefinite programming.
#
#
# References
# ----------
#
# .. footbibliography::
