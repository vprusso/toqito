"""
# Calculating the quantum and classical value of a two-player XOR game

In this tutorial, we will cover the concept of an *XOR game*. We will also
showcase how the `|toqito⟩` software package can be used to calculate the
classical and quantum value of a given XOR game.
"""

# %%
# For readers who are already familiar with XOR games and who simply want to see
# how to use `|toqito⟩` to study these objects, they are welcome to consult the
# documentation page, and more specifically the function
# [xor_game_value](../).
#
# Further information beyond the scope of this tutorial on the notion of XOR
# games along with the method of computing their quantum value may be found in
# [@Cleve_2008_Strong].
#
# ## Two-player XOR games
#
# A two-player XOR game is a nonlocal game in which the winning condition is
# predicated on an XOR function. For more information on the more general class
# of nonlocal games along with how one defines classical and quantum strategies
# for these games, please refer to the example in [Nonlocal games](../nonlocal_game)
#
# !!! note
#
#     It is *not* known how to directly compute the quantum value of an arbitrary
#     nonlocal game. For the subset of XOR games, it turns out that it is
#     possible to directly calculate the quantum value by solving a semidefinite
#     program. The `|toqito⟩` package obtains the quantum value of an XOR game
#     in this manner.
#
# The rest of this tutorial is concerned with analyzing specific XOR games.
#
# ## The CHSH game
#
# The *CHSH game* is a two-player XOR game with the following probability
# distribution and question and answer sets.
#
# $$
# \begin{equation}
# \begin{aligned}
# \pi(x,y) = \frac{1}{4}, \qquad (x,y) \in \Sigma_A \times \Sigma_B, \qquad \text{and} \qquad (a, b) \in \Gamma_A \times \Gamma_B,
# \end{aligned}
# \end{equation}
# $$
#
# where
#
# $$
# \begin{equation}
# \Sigma_A = \{0, 1\}, \quad \Sigma_B = \{0, 1\}, \quad \Gamma_A = \{0,1\}, \quad \text{and} \quad \Gamma_B = \{0, 1\}.
# \end{equation}
# $$
#
# Alice and Bob win the CHSH game if and only if the following equation is
# satisfied
#
# $$
# \begin{equation}
# a \oplus b = x y.
# \end{equation}
# $$
#
# Recall that $\oplus$ refers to the XOR operation.
#
# For each question scenario, the following table provides what the winning
# condition must be equal to for each question tuple to induce a winning outcome.
#
# | $x$   | $y$   |  $a \oplus b$  |
# |-------|-------|----------------|
# | $0$   | $0$   | $0$            |
# | $0$   | $1$   | $0$            |
# | $1$   | $0$   | $0$            |
# | $1$   | $1$   | $1$            |
#
# In order to specify an XOR game in `|toqito⟩`, we will define two matrices:
#
# * `prob_mat`: A matrix whose $(x, y)^{th}$ entry corresponds to
#   the probability that Alice receives question $x$ and Bob receives
#   question $y$.
#
# * `pred_mat`: A matrix whose $(x, y)^{th}$ entry corresponds to
#   the winning choice of $a$ and $b$ when Alice receives
#   $x$ and Bob receives $y$ from the referee.
#
#
# For the CHSH game, the `prob_mat` and `pred_mat` variables are defined as follows.
#
import numpy as np

prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
pred_mat = np.array([[0, 0], [0, 1]])

# %%
# That is, the `prob_mat` matrix encapsulates that each question pair
# $\{(0,0), (0, 1), (1, 0), (1, 1)\}$ is equally likely.
#
# The `pred_mat` matrix indicates what the winning outcome of Alice and Bob
# should be. For instance, `pred_mat[0][0] = 0` describes the scenario where
# Alice and Bob both receive $0$ as input. As we want to satisfy the
# winning condition $x \land y = a \oplus b$, we must have that $a
# \oplus b = 0$ to satisfy the case when both $x$ and $y$ are equal
# to zero. A similar logic can be followed to populate the remaining entries of
# the `pred_mat` variable.
#
# We will use both of the `prob_mat` and `pred_mat` variables in the
# coming subsections to make use of the `|toqito⟩` package to compute both the
# classical and quantum value of the CHSH game.
#
# ### A classical strategy for the CHSH game
#
# We can begin by asking; is it possible for Alice and Bob to win for every
# single question pair they receive with certainty? If Alice and Bob use a
# classical strategy, the answer to this question is "no". To see why, consider
# the following equations:
#
# $$
# \begin{equation}
# \begin{aligned}
# a_0 \oplus b_0 = 0, \quad a_0 \oplus b_1 = 0, \\
# a_1 \oplus b_0 = 0, \quad a_1 \oplus b_1 = 1.
# \end{aligned}
# \end{equation}
# $$
#
# In the above equation, $a_x$ is Alice's answer in the event that she
# receives question $x$ from the referee for $x \in \Sigma_A$.
# Similarly, $b_y$ is Bob's answer when Bob receives question $y$
# from the referee for $y \in \Sigma_B$. These equations express the
# winning conditions that Alice and Bob must satisfy in order to perfectly win
# the CHSH game. That is, if it's possible to satisfy all of these equations
# simultaneously, it's not possible for them to lose.
#
# One could perform a brute-force check to see that there is no possible way for
# Alice and Bob to simultaneously satisfy all four equations. The best they can
# do is satisfy three out of the four equations
#
# $$
# \begin{equation}
# \begin{aligned}
# a_0 \oplus b_0 = 0, \quad a_0 \oplus b_1 = 0, \\
# a_1 \oplus b_0 = 0.
# \end{aligned}
# \end{equation}
# $$
#
# They can achieve this if they either have answers $a_0 = b_0 = a_1 = b_1
# = 0$ or $a_0 = b_0 = a_1 = b_1 = 1$.
#
# Since it is not possible to satisfy all four equations, but it is possible to
# satisfy three out of the four equations, the classical value of the CHSH game
# is $3/4$, or stated in an equivalent way
#
# $$
# \begin{equation}
# \omega(G_{CHSH}) = 3/4 = 0.75.
# \end{equation}
# $$
#
# We can verify this by making use of `|toqito⟩` to compute the classical
# value of the CHSH game.
#
from toqito.nonlocal_games.xor_game import XORGame

chsh = XORGame(prob_mat, pred_mat)
print(f"The classical value of game is: {chsh.classical_value()}")

# %%
# ### A quantum strategy for the CHSH game
#
# What is very intriguing about the CHSH game is that it is an example of a
# nonlocal game where the players can do *strictly better* if they make use of a
# quantum strategy instead of a classical one. The quantum strategy that allows
# the players to do strictly better is composed of the following shared state and
# sets of measurements.
#
# * State: The players prepare and share the state:
#
# $$
# | \psi \rangle = \frac{1}{\sqrt{2}}
# \left(| 00 \rangle + | 11 \rangle \right).
# $$
#
# * Measurements: The players measure with respect to the following basis
#
# $$
# | \phi_0 \rangle = \cos(\theta)|0 \rangle + \sin(\theta)|1 \rangle, \quad
# | \phi_1 \rangle = -\sin(\theta)|0 \rangle + \cos(\theta)|1 \rangle,
# $$
#
#   such that
#
#   * If $x = 0$ Alice sets $\theta = 0$.
#     Otherwise, if $x = 1$, Alice sets $\theta = \pi/4$.
#
#   * If $y = 0$ Bob sets $\theta = \pi/8$.
#     Otherwise, if $y = 1$, Bob sets $\theta = -\pi/8$.
#
#
# We can now analyze how well this particular quantum strategy performs by
# analyzing what occurs in each of the four possible scenarios. For brevity, we
# will just analyze the first case, but analyzing the remaining cases follows a
# similar analysis.
#
# * Case: $x = 0, y = 0$:
#
#   In this case, Alice and Bob win if $a = b = 0$ or if $a = b = 1$.
#   Alice receives question $x$ and selects her measurements constructed from
#   the basis as specified in the strategy.
#
# $$
# A_0^0 = | \phi_0 \rangle \langle \phi_0 |
# \quad \text{and} \quad
# A_1^0 = | \phi_1 \rangle \langle \phi_1 |
# $$
#
#   where
#
# $$
# \begin{aligned}
# | \phi_0 \rangle &= \cos(0)| 0 \rangle + \sin(0)| 1 \rangle, \\
# | \phi_1 \rangle &= -\sin(0)| 0 \rangle + \cos(0)| 1 \rangle.
# \end{aligned}
# $$
#
#   In a similar way, since Bob receives question $y = 0$, he selects his
#   measurements from the basis
#
# $$
# \begin{aligned}
# | \phi_0 \rangle &= \cos(\pi/8)| 0 \rangle + \sin(\pi/8)| 1 \rangle, \\
# | \phi_1 \rangle &= -\sin(\pi/8)| 0 \rangle + \cos(\pi/8)| 1 \rangle.
# \end{aligned}
# $$
#
#   where the measurement operators themselves are defined as
#
# $$
# B_0^0 = | \phi_0 \rangle
# \quad \text{and} \quad
# B_1^0 = | \phi_1 \rangle \langle \phi_1 |
# $$
#
#   Using these measurements, we can calculate the probability that Alice and Bob
#   win on the inputs $x = 0$ and $y = 0$ as
#
# $$
# p(a, b|0, 0) = \langle \psi | A_0^0 \otimes B_0^0 | \psi \rangle +
# \langle \psi | A_1^0 \otimes B_1^0 | \psi \rangle.
# $$
#
# Calculating the above equation and normalizing by a factor of $1/4$, we
# obtain the value of $\cos^2(\pi/8)$. Calculating the remaining three
# cases of $(x = 0, y = 1), (x = 1, y = 0)$, and $(x = 1, y = 1)$
# follow a similar analysis.
#
# We can see that using this quantum strategy the players win the CHSH game with
# a probability of $\cos^2(\pi/8) \approx 0.85355$, which is quite a bit
# better than the best classical strategy yielding a probability of $3/4$
# to win. As it turns out, the winning probability $\cos^2(\pi/8)$ using a
# quantum strategy is optimal, which we can represent as
# $\omega^*(G_{CHSH}) = \cos^2(\pi/8)$.
#
# We can calculate the quantum value of the CHSH game using `|toqito⟩` as
# follows:

import numpy as np

print(f"The quantum value of game is: {np.around(chsh.quantum_value(), decimals=2)}")


# %%
# For reference, the complete code to calculate both the classical and quantum
# values of the CHSH game is provided below.

import numpy as np

from toqito.nonlocal_games.xor_game import XORGame

prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
pred_mat = np.array([[0, 0], [0, 1]])
chsh = XORGame(prob_mat, pred_mat)
print(f"The classical value of games is: {chsh.classical_value()}")
print(f"The quantum value of games is: {np.around(chsh.quantum_value(), decimals=2)}")

# %%
# ## The odd cycle game
#
# The *odd cycle game* is another two-player XOR game with the following question and answer sets
#
# $$
# \begin{equation}
# \begin{aligned}
# \Sigma_{A} = \Sigma_B = \mathbb{Z}_n \qquad \text{and} \qquad \Gamma_A = \Gamma_B = \{0, 1\},
# \end{aligned}
# \end{equation}
# $$
#
# where $\pi$ is the uniform probability distribution over the question set.
#
# As an example, we can specify the odd cycle game for $n=5$ and calculate
# the classical and quantum values of this game.
#
import numpy as np

from toqito.nonlocal_games.xor_game import XORGame

# Define the probability matrix.
prob_mat = np.array(
    [[0.1, 0.1, 0, 0, 0], [0, 0.1, 0.1, 0, 0], [0, 0, 0.1, 0.1, 0], [0, 0, 0, 0.1, 0.1], [0.1, 0, 0, 0, 0.1]]
)

# Define the predicate matrix.
pred_mat = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0]])

# Compute the classical and quantum values.
odd_cycle = XORGame(prob_mat, pred_mat)
print(f"The classical value of games is: {np.around(odd_cycle.classical_value(), decimals=2)}")
print(f"The quantum value of games is: {np.around(odd_cycle.quantum_value(), decimals=2)}")
# mkdocs_gallery_thumbnail_path = 'figures/logo.png'

# %%
# Note that the odd cycle game is another example of an XOR game where the
# players are able to win with a strictly higher probability if they adopt a
# quantum strategy. For a general XOR game, Alice and Bob may perform equally
# well whether they adopt either a quantum or classical strategy. It holds that
# the quantum value for any XOR game is a natural upper bound on the classical
# value. That is, for an XOR game, $G$, it holds that
#
# $$
# \omega(G) \leq \omega^*(G),
# $$
#
# for every XOR game $G$.
#