# ruff: noqa: D205, D400, D415
"""CHSH Game Example
=================

This example calculates the classical and quantum values for the CHSH game
using the toqito library. It demonstrates how to construct the necessary matrices
and use the XORGame function.
"""

# sphinx_gallery_thumbnail_path = '_static/chsh_game.png'

import numpy as np

from toqito.nonlocal_games.xor_game import XORGame

# %%
# Introduction to the CHSH Game
# -----------------------------
#
# The CHSH game involves two players, Alice and Bob, who cannot communicate once
# the game begins. They each receive a random bit (0 or 1) and must output a bit
# without communicating. They win if the XOR of their outputs equals the AND of
# their inputs.
#
# Mathematically, if Alice gets input x and outputs a, while Bob gets input y and
# outputs b, they win if:
#
# .. math::
#    a \oplus b = x \land y
#
# where :math:`\oplus` represents XOR (addition modulo 2) and :math:`\land` represents
# logical AND.

# %%
# Setting Up the Game Parameters
# ------------------------------
#
# To define the CHSH game, we need two matrices:
#
# 1. A probability matrix that specifies the distribution of inputs to the players
# 2. A predicate matrix that encodes the winning condition
#
# For the CHSH game, inputs are uniformly distributed, so each pair (x,y) has
# probability 1/4:

# The probability distribution matrix for inputs (x,y)
prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])

print("Probability matrix (distribution of inputs):")
print(prob_mat)

# %%
# Next, we define the predicate matrix. For XOR games in toqito, this is a matrix
# where each entry V(x,y) = (-1)^(desired answer for x,y). In the CHSH game,
# the desired answer is x AND y.

# Predicate matrix where entries are V(x,y) = (-1)^(x∧y)
pred_mat = np.zeros((2, 2))
for x in range(2):
    for y in range(2):
        # For CHSH: winning condition is a ⊕ b = x ∧ y
        # Expressed as V(x,y) = (-1)^(x∧y)
        pred_mat[x, y] = (-1) ** (x & y)

print("\nPredicate matrix (encoding the winning condition):")
print(pred_mat)

# %%
# Computing the Classical Value
# -----------------------------
#
# The classical value of a nonlocal game is the maximum probability of winning
# when players use classical strategies (shared randomness, but no quantum
# entanglement).
#
# For the CHSH game, theoretical analysis shows this value is 0.75 (or 3/4).
# Let's compute it using toqito:

# Create an XORGame object and calculate the classical value
chsh_game = XORGame(prob_mat, pred_mat)
classical_val = chsh_game.classical_value()
print(f"Classical value of the CHSH game: {classical_val}")

# %%
# Computing the Quantum Value
# ---------------------------
#
# The quantum value represents the maximum probability of winning when players
# can use quantum strategies, such as sharing an entangled quantum state.
#
# For the CHSH game, the quantum value is :math:`\cos^2(\pi/8) \approx 0.85`,
# which exceeds the classical bound of :math:`0.75`.
#
# The CHSH inequality can be expressed mathematically as:
#
# .. math::
#    |E(0,0) + E(0,1) + E(1,0) - E(1,1)| \leq 2
#
# Where :math:`E(x,y)` represents the expected value of the product of Alice's
# and Bob's outputs when given inputs :math:`x` and :math:`y` respectively.
#
# Quantum mechanics predicts this value can reach :math:`2\sqrt{2} \approx 2.82`,
# which is known as Tsirelson's bound.

quantum_val = chsh_game.quantum_value()
print(f"Quantum value of the CHSH game: {quantum_val}")
print(f"Quantum advantage: {quantum_val - classical_val}")

# %%
# Optimal Quantum Strategy
# ------------------------
#
# The optimal quantum strategy for the CHSH game involves:
#
# 1. Sharing a maximally entangled state (Bell state): :math:`|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}`
# 2. Using specific measurements based on the inputs:
#    * For input 0, Alice measures in the Z basis
#    * For input 1, Alice measures in the X basis
#    * For input 0, Bob measures in the :math:`\frac{Z+X}{\sqrt{2}}` basis
#    * For input 1, Bob measures in the :math:`\frac{Z-X}{\sqrt{2}}` basis
#
# These measurements are chosen to maximize the quantum value of the CHSH game.
# The resulting correlation allows the players to win with probability :math:`\cos^2(\pi/8) \approx 0.85`.

# %%
# Conclusion
# ----------
#
# This example demonstrates the quantum advantage in the CHSH game:
#
# * The classical value is 0.75 (75% winning probability)
# * The quantum value is approximately 0.85 (85% winning probability)
# * This represents a clear quantum advantage of about 10%
#
# The CHSH game has profound implications for our understanding of quantum mechanics
# and the nature of reality. The ability of quantum strategies to exceed classical
# bounds shows that quantum mechanics cannot be explained by any local hidden
# variable theory, as famously argued by Einstein, Podolsky, and Rosen.
#
# This quantum advantage has been verified experimentally multiple times, confirming
# the non-local nature of quantum mechanics as described by Bell's theorem.
