"""The CHSH extended nonlocal game
=====================================

Following our analysis of the BB84 game, let us now define another important
extended nonlocal game, :math:`G_{CHSH}`. This game is defined by a winning
condition reminiscent of the standard CHSH nonlocal game.

"""

# %%
# Let :math:`\Sigma_A = \Sigma_B = \Gamma_A = \Gamma_B = \{0,1\}`, define a
# collection of measurements :math:`\{V(a,b|x,y) : a \in \Gamma_A, b \in
# \Gamma_B, x \in \Sigma_A, y \in \Sigma_B\} \subset \text{Pos}(\mathcal{R})`
# such that
#
# .. math::
#    \begin{equation}
#        \begin{aligned}
#            V(0,0|0,0) = V(0,0|0,1) = V(0,0|1,0) = \begin{pmatrix}
#                                                    1 & 0 \\
#                                                    0 & 0
#                                                   \end{pmatrix}, \\
#            V(1,1|0,0) = V(1,1|0,1) = V(1,1|1,0) = \begin{pmatrix}
#                                                    0 & 0 \\
#                                                    0 & 1
#                                                   \end{pmatrix}, \\
#            V(0,1|1,1) = \frac{1}{2}\begin{pmatrix}
#                                        1 & 1 \\
#                                        1 & 1
#                                    \end{pmatrix}, \\
#            V(1,0|1,1) = \frac{1}{2} \begin{pmatrix}
#                                        1 & -1 \\
#                                        -1 & 1
#                                     \end{pmatrix},
#        \end{aligned}
#    \end{equation}
#
# define
#
# .. math::
#    V(a,b|x,y) = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}
#
# for all :math:`a \oplus b \not= x \land y`, and define :math:`\pi(0,0) =
# \pi(0,1) = \pi(1,0) = \pi(1,1) = 1/4`.
#
# In the event that :math:`a \oplus b \not= x \land y`, the referee's measurement
# corresponds to the zero matrix. If instead it happens that :math:`a \oplus b =
# x \land y`, the referee then proceeds to measure with respect to one of the
# measurement operators. This winning condition is reminiscent of the standard
# CHSH nonlocal game.
#
# We can encode :math:`G_{CHSH}` in a similar way using :code:`numpy` arrays as
# we did for :math:`G_{BB84}`.

# Define the CHSH extended nonlocal game.
import numpy as np

# The dimension of referee's measurement operators:
dim = 2
# The number of outputs for Alice and Bob:
a_out, b_out = 2, 2
# The number of inputs for Alice and Bob:
a_in, b_in = 2, 2

# Define the predicate matrix V(a,b|x,y) \in Pos(R)
chsh_pred_mat = np.zeros([dim, dim, a_out, b_out, a_in, b_in])

# V(0,0|0,0) = V(0,0|0,1) = V(0,0|1,0).
chsh_pred_mat[:, :, 0, 0, 0, 0] = np.array([[1, 0], [0, 0]])
chsh_pred_mat[:, :, 0, 0, 0, 1] = np.array([[1, 0], [0, 0]])
chsh_pred_mat[:, :, 0, 0, 1, 0] = np.array([[1, 0], [0, 0]])

# V(1,1|0,0) = V(1,1|0,1) = V(1,1|1,0).
chsh_pred_mat[:, :, 1, 1, 0, 0] = np.array([[0, 0], [0, 1]])
chsh_pred_mat[:, :, 1, 1, 0, 1] = np.array([[0, 0], [0, 1]])
chsh_pred_mat[:, :, 1, 1, 1, 0] = np.array([[0, 0], [0, 1]])

# V(0,1|1,1)
chsh_pred_mat[:, :, 0, 1, 1, 1] = 1 / 2 * np.array([[1, 1], [1, 1]])

# V(1,0|1,1)
chsh_pred_mat[:, :, 1, 0, 1, 1] = 1 / 2 * np.array([[1, -1], [-1, 1]])

# The probability matrix encode \pi(0,0) = \pi(0,1) = \pi(1,0) = \pi(1,1) = 1/4.
chsh_prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])

# %%
# Example: The unentangled value of the CHSH extended nonlocal game
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Similar to what we did for the BB84 extended nonlocal game, we can also compute
# the unentangled value of :math:`G_{CHSH}`.

# Calculate the unentangled value of the CHSH extended nonlocal game
import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

# Define an ExtendedNonlocalGame object based on the CHSH game.
chsh = ExtendedNonlocalGame(chsh_prob_mat, chsh_pred_mat)

# The unentangled value is 3/4 = 0.75
print("The unentangled value is ", np.around(chsh.unentangled_value(), decimals=2))

# %%
# We can also run multiple repetitions of :math:`G_{CHSH}`.

# The unentangled value of CHSH under parallel repetition.
import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

# Define the CHSH game for two parallel repetitions.
chsh_2_reps = ExtendedNonlocalGame(chsh_prob_mat, chsh_pred_mat, 2)

# The unentangled value for two parallel repetitions is (3/4)**2 \approx 0.5625
print("The unentangled value for two parallel repetitions is ", np.around(chsh_2_reps.unentangled_value(), decimals=2))

# %%
# Note that strong parallel repetition holds as
#
# .. math::
#    \omega(G_{CHSH})^2 = \omega(G_{CHSH}^2) = \left(\frac{3}{4}\right)^2.
#
# Example: The non-signaling value of the CHSH extended nonlocal game
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To obtain an upper bound for :math:`G_{CHSH}`, we can calculate the
# non-signaling value.

# Calculate the non-signaling value of the CHSH extended nonlocal game.
import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

# Define an ExtendedNonlocalGame object based on the CHSH game.
chsh = ExtendedNonlocalGame(chsh_prob_mat, chsh_pred_mat)

# The non-signaling value is 3/4 = 0.75
print("The non-signaling value is ", np.around(chsh.nonsignaling_value(), decimals=2))

# %%
# As we know that :math:`\omega(G_{CHSH}) = \omega_{ns}(G_{CHSH}) = 3/4` and that
#
# .. math::
#    \omega(G) \leq \omega^*(G) \leq \omega_{ns}(G)
#
# for any extended nonlocal game, :math:`G`, we may also conclude that
# :math:`\omega^*(G) = 3/4`.

# %%
# As we know that :math:`\omega(G_{CHSH}) = \omega_{ns}(G_{CHSH}) = 3/4` and that
#
# .. math::
#    \omega(G) \leq \omega^*(G) \leq \omega_{ns}(G)
#
# for any extended nonlocal game, :math:`G`, we may also conclude that
# :math:`\omega^*(G_{CHSH}) = 3/4`.
#
# So far, both the BB84 and CHSH examples have demonstrated cases where the
# unentangled and standard quantum values are equal. In the next tutorial, :ref:`sphx_glr_auto_examples_extended_nonlocal_games_example_enlg_mub.py` we
# will explore a game based on mutually unbiased bases that exhibits a strict
# quantum advantage, where :math:`\omega(G) < \omega^*(G)`.

# %%
# References
# ----------
#
# .. footbibliography::
