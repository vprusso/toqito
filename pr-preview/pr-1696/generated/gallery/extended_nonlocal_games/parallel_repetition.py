"""# Parallel Repetition of Extended Nonlocal Games

Explores parallel repetition for extended nonlocal games, demonstrating
that strong parallel repetition holds for unentangled strategies but
fails for non-signaling strategies. Uses the BB84 extended nonlocal
game as the primary example.
"""

# %%
# ## What is parallel repetition?
#
# Given a nonlocal game $G$, the *$r$-fold parallel repetition* $G^r$
# is a new game where $r$ independent copies of $G$ are played
# simultaneously. Alice receives $r$ questions $(x_1, \ldots, x_r)$
# and must provide $r$ answers $(a_1, \ldots, a_r)$, and similarly for
# Bob. The players win $G^r$ if and only if they win *every* individual
# round.
#
# A natural question is whether the optimal winning probability
# decreases multiplicatively with $r$:
#
# $$
# \omega(G^r) \stackrel{?}{=} \omega(G)^r
# $$
#
# If this equality holds, we say that $G$ satisfies *strong parallel
# repetition*. If $\omega(G) < 1$, strong parallel repetition implies
# the winning probability decreases exponentially in $r$, making
# cheating progressively harder — a property with important
# consequences for cryptographic protocols.
#
# ## Parallel repetition in `|toqito⟩`
#
# The `ExtendedNonlocalGame` class supports parallel repetition via its
# `reps` parameter. Under the hood, this constructs the tensor product
# of the probability distribution and prediction operators:
#
# ```python
# # Single game
# game = ExtendedNonlocalGame(prob_mat, pred_mat)
#
# # r-fold parallel repetition
# game_r = ExtendedNonlocalGame(prob_mat, pred_mat, reps=r)
# ```
#
# ## Example: The BB84 extended nonlocal game
#
# The BB84 extended nonlocal game is based on the two bases used in the
# BB84 quantum key distribution protocol: the computational basis
# $\{|0\rangle, |1\rangle\}$ and the Hadamard basis
# $\{|+\rangle, |-\rangle\}$.
#
# For the single game, all three values coincide:
#
# $$
# \omega(G_{BB84}) = \omega^*(G_{BB84}) = \omega_{ns}(G_{BB84}) = \cos^2(\pi/8) \approx 0.854.
# $$
#
# Let us first set up the BB84 game.

# %%
import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

# Referee's dimension (qubit).
dim_referee = 2

# Prediction operators for the BB84 game.
e_0, e_1 = np.array([[1], [0]]), np.array([[0], [1]])
e_p = (e_0 + e_1) / np.sqrt(2)
e_m = (e_0 - e_1) / np.sqrt(2)

bb84_pred_mat = np.zeros([2, 2, 2, 2, dim_referee, dim_referee], dtype=complex)
bb84_pred_mat[0, 0, 0, 0] = e_0 @ e_0.conj().T
bb84_pred_mat[1, 1, 0, 0] = e_1 @ e_1.conj().T
bb84_pred_mat[0, 0, 1, 1] = e_p @ e_p.conj().T
bb84_pred_mat[1, 1, 1, 1] = e_m @ e_m.conj().T

bb84_prob_mat = np.array([[0.5, 0], [0, 0.5]])

bb84 = ExtendedNonlocalGame(bb84_prob_mat, bb84_pred_mat)

# %%
# ### Single game values
#
# Compute the unentangled and non-signaling values for the single game.

# %%
omega_ue = bb84.unentangled_value()
omega_ns = bb84.nonsignaling_value()
exact = np.cos(np.pi / 8) ** 2

print(f"Unentangled value:    {omega_ue:.5f}")
print(f"Non-signaling value:  {omega_ns:.5f}")
print(f"Exact (cos²(π/8)):    {exact:.5f}")

# %%
# ### Strong parallel repetition for unentangled strategies
#
# For unentangled (classical) strategies, strong parallel repetition
# *does* hold. The unentangled value of the 2-fold repetition should
# equal the square of the single-game value:
#
# $$
# \omega(G_{BB84}^2) = \omega(G_{BB84})^2 = \cos^4(\pi/8) \approx 0.729.
# $$

# %%
bb84_2_reps = ExtendedNonlocalGame(bb84_prob_mat, bb84_pred_mat, reps=2)

omega_ue_2 = bb84_2_reps.unentangled_value()
expected_ue_2 = exact**2

print(f"ω(G²) unentangled:   {omega_ue_2:.5f}")
print(f"ω(G)² expected:      {expected_ue_2:.5f}")
print(f"Strong parallel rep holds: {np.isclose(omega_ue_2, expected_ue_2, atol=1e-2)}")

# %%
# ### Failure of strong parallel repetition for non-signaling strategies
#
# In contrast, the non-signaling value does *not* satisfy strong
# parallel repetition. As shown in [@cosentino2015quantum], the
# non-signaling value of $G_{BB84}^2$ is strictly greater than
# $\omega_{ns}(G_{BB84})^2$:
#
# $$
# \omega_{ns}(G_{BB84}^2) \approx 0.738 > 0.729 \approx \omega_{ns}(G_{BB84})^2.
# $$
#
# This demonstrates that non-signaling strategies can exploit
# correlations across parallel rounds in a way that classical strategies
# cannot.

# %%
omega_ns_2 = bb84_2_reps.nonsignaling_value()
expected_ns_2 = exact**2

print(f"ω_ns(G²):            {omega_ns_2:.5f}")
print(f"ω_ns(G)²:            {expected_ns_2:.5f}")
print(f"ω_ns(G²) > ω_ns(G)²: {omega_ns_2 > expected_ns_2 + 1e-3}")

# %%
# ## Implications
#
# The failure of strong parallel repetition for non-signaling strategies
# has significant implications:
#
# 1. **Cryptographic security**: Protocols that rely on parallel
#    repetition to amplify security gaps must carefully account for
#    which strategy class the adversary has access to.
#
# 2. **Separation of strategy classes**: While unentangled and
#    non-signaling values coincide for the single BB84 game, they
#    diverge under parallel repetition, revealing a fundamental
#    structural difference.
#
# 3. **Quantum vs. non-signaling**: The quantum value under parallel
#    repetition remains an open question for many extended nonlocal
#    games, including BB84.
#
# For further details, see [@cosentino2015quantum].

# %%
# mkdocs_gallery_thumbnail_path = 'figures/extended_nonlocal_game.svg'
