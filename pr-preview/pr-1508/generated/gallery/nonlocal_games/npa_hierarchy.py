"""# The NPA Hierarchy

An introduction to the Navascues-Pironio-Acin (NPA) hierarchy for
bounding the quantum value of nonlocal games via semidefinite
programming. Covers the theory, demonstrates usage in toqito for the
CHSH game and odd-cycle games, and shows how to use intermediate
hierarchy levels.
"""

# %%
# ## Overview
#
# Computing the quantum value of a nonlocal game is in general undecidable.
# However, the **NPA hierarchy** [@navascues2008convergent] provides a
# sequence of semidefinite programs (SDPs) that yield increasingly tight
# *upper bounds* on the quantum value. As the level of the hierarchy
# increases, these bounds converge to the true quantum (commuting operator)
# value.
#
# The hierarchy works by constructing a **moment matrix** $\Gamma$ whose
# entries correspond to expectation values of products of measurement
# operators. At level $k$, the matrix includes all products of up to $k$
# measurement operators. The key insight is that any valid quantum
# strategy must produce a moment matrix that is positive semidefinite.
#
# For a nonlocal game with predicate $V$ and probability distribution
# $\pi$, the quantum value is bounded by:
#
# $$
# \omega^*(G) \leq \omega_k^{\text{NPA}}(G)
# $$
#
# where $\omega_k^{\text{NPA}}(G)$ is the optimal value of the level-$k$
# SDP relaxation.
#
# !!! note
#
#     For more details, see the original paper
#     [@navascues2008convergent] as well as the lecture notes by
#     Watrous in [@watrous2018theory]. The hierarchy can also be
#     generalized to extended nonlocal games, as described in
#     [@cosentino2015quantum].
#
# ## Using the NPA hierarchy in `|toqito⟩`
#
# The `|toqito⟩` package provides the NPA hierarchy through
# the `NonlocalGame` class via its `commuting_measurement_value_upper_bound`
# method. This method accepts a parameter `k` that specifies the level
# of the hierarchy.
#
# The level can be:
#
# - A **positive integer** (e.g., `k=1`, `k=2`): uses all operator
#   products up to that length.
# - A **string** (e.g., `k="1+ab"`): specifies intermediate levels.
#   For example, `"1+ab"` uses level 1 plus all products of one Alice
#   and one Bob measurement operator.
#
# ## Example: The CHSH game
#
# The CHSH game is a canonical example where the quantum value is known
# exactly: $\omega^*(G_{\text{CHSH}}) = \cos^2(\pi/8) \approx 0.8536$.
#
# Let's compute upper bounds on the quantum value at different levels of
# the NPA hierarchy and see how they converge.

# %%
import numpy as np
from toqito.nonlocal_games.nonlocal_game import NonlocalGame

# Define the CHSH game
dim = 2
num_alice_inputs, num_alice_outputs = 2, 2
num_bob_inputs, num_bob_outputs = 2, 2

prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
pred_mat = np.zeros((num_alice_outputs, num_bob_outputs, num_alice_inputs, num_bob_inputs))

for a_alice in range(num_alice_outputs):
    for b_bob in range(num_bob_outputs):
        for x_alice in range(num_alice_inputs):
            for y_bob in range(num_bob_inputs):
                if np.mod(a_alice + b_bob + x_alice * y_bob, dim) == 0:
                    pred_mat[a_alice, b_bob, x_alice, y_bob] = 1

chsh = NonlocalGame(prob_mat, pred_mat)

# %%
# ### Level 1
#
# At level 1, the NPA hierarchy provides the loosest upper bound.

# %%
npa_level_1 = chsh.commuting_measurement_value_upper_bound(k=1)
print(f"NPA level 1 upper bound: {npa_level_1:.6f}")

# %%
# ### Intermediate level: "1+ab"
#
# Using `k="1+ab"` adds products of one Alice and one Bob operator to
# the level-1 moment matrix, giving a tighter bound without the full
# cost of level 2.

# %%
npa_level_1ab = chsh.commuting_measurement_value_upper_bound(k="1+ab")
print(f"NPA level 1+ab upper bound: {npa_level_1ab:.6f}")
print(f"Known quantum value:        {np.cos(np.pi / 8) ** 2:.6f}")

# %%
# For the CHSH game, even the `"1+ab"` level already provides a value
# very close to the true quantum value of $\cos^2(\pi/8)$.
#
# ## Example: The odd-cycle game
#
# The odd-cycle game on $n$ vertices is another well-studied nonlocal
# game. The quantum value is known to be:
#
# $$
# \omega^*(G_n) = \frac{1}{2} + \frac{1}{2} \cos\!\left(\frac{\pi}{n}\right)
# $$
#
# Let's compute the NPA bound for the 5-cycle game.

# %%
n = 5  # Number of vertices in the odd cycle

# Probability matrix: uniform over edges
prob_mat_cycle = np.zeros((n, n))
for i in range(n):
    prob_mat_cycle[i, (i + 1) % n] = 1 / n

# Predicate: Alice and Bob win if their outputs are equal (for non-wrap edges)
# or different (for the wrap-around edge)
pred_mat_cycle = np.zeros((2, 2, n, n))
for x in range(n):
    y = (x + 1) % n
    for a in range(2):
        for b in range(2):
            if x < n - 1:
                # Non-wrap edge: win if a == b
                if a == b:
                    pred_mat_cycle[a, b, x, y] = 1
            else:
                # Wrap-around edge: win if a != b
                if a != b:
                    pred_mat_cycle[a, b, x, y] = 1

odd_cycle = NonlocalGame(prob_mat_cycle, pred_mat_cycle)

# Compute quantum value via NPA
npa_val = odd_cycle.commuting_measurement_value_upper_bound(k="1+ab")
exact_val = 0.5 + 0.5 * np.cos(np.pi / n)

print(f"NPA upper bound (5-cycle):  {npa_val:.6f}")
print(f"Known quantum value:        {exact_val:.6f}")

# %%
# ## Convergence of the hierarchy
#
# A key property of the NPA hierarchy is that as $k \to \infty$, the
# upper bounds converge to the commuting operator value:
#
# $$
# \omega_1^{\text{NPA}} \geq \omega_2^{\text{NPA}} \geq \cdots
# \geq \omega^*(G)
# $$
#
# In practice, low levels of the hierarchy (1 or 2) often suffice to
# get close to the true value for well-structured games. Higher levels
# provide tighter bounds but at increased computational cost, as the
# size of the moment matrix grows polynomially in the number of
# measurement operators.

# %%
# mkdocs_gallery_thumbnail_path = 'figures/nonlocal_game.svg'
