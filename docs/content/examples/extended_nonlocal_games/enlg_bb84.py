"""# The BB84 extended nonlocal game

Constructs the BB84 extended nonlocal game and uses toqito to calculate its
unentangled, standard quantum, and non-signaling values. Demonstrates that all
three values coincide and that strong parallel repetition holds for the
unentangled value but fails in the non-signaling setting.
"""

# %%
# The *BB84 extended nonlocal game* is defined as follows. Let $\Sigma_A =
# \Sigma_B = \Gamma_A = \Gamma_B = \{0,1\}$, define
#
# $$
# \begin{equation}
# \begin{aligned}
# V(0,0|0,0) = \begin{pmatrix}
# 1 & 0 \\
# 0 & 0
# \end{pmatrix}, &\quad
# V(1,1|0,0) = \begin{pmatrix}
# 0 & 0 \\
# 0 & 1
# \end{pmatrix}, \\
# V(0,0|1,1) = \frac{1}{2}\begin{pmatrix}
# 1 & 1 \\
# 1 & 1
# \end{pmatrix}, &\quad
# V(1,1|1,1) = \frac{1}{2}\begin{pmatrix}
# 1 & -1 \\
# -1 & 1
# \end{pmatrix},
# \end{aligned}
# \end{equation}
# $$
#
# define
#
# $$
# V(a,b|x,y) = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}
# $$
#
# for all $a \not= b$ or $x \not= y$, define $\pi(0,0) =
# \pi(1,1) = 1/2$, and define $\pi(x,y) = 0$ if $x \not=y$.
#
# We can encode the BB84 game, $G_{BB84} = (\pi, V)$, in `numpy`
# arrays where `prob_mat` corresponds to the probability distribution
# $\pi$ and where `pred_mat` corresponds to the operator $V$.
#

# Define the BB84 extended nonlocal game.
import numpy as np

from toqito.states import basis

# The basis: {|0>, |1>}:
e_0, e_1 = basis(2, 0), basis(2, 1)

# The basis: {|+>, |->}:
e_p = (e_0 + e_1) / np.sqrt(2)
e_m = (e_0 - e_1) / np.sqrt(2)

# The dimension of referee's measurement operators:
dim = 2
# The number of outputs for Alice and Bob:
a_out, b_out = 2, 2
# The number of inputs for Alice and Bob:
a_in, b_in = 2, 2

# Define the predicate matrix V(a,b|x,y) \in Pos(R)
bb84_pred_mat = np.zeros([dim, dim, a_out, b_out, a_in, b_in])

# V(0,0|0,0) = |0><0|
bb84_pred_mat[:, :, 0, 0, 0, 0] = e_0 @ e_0.conj().T
# V(1,1|0,0) = |1><1|
bb84_pred_mat[:, :, 1, 1, 0, 0] = e_1 @ e_1.conj().T
# V(0,0|1,1) = |+><+|
bb84_pred_mat[:, :, 0, 0, 1, 1] = e_p @ e_p.conj().T
# V(1,1|1,1) = |-><-|
bb84_pred_mat[:, :, 1, 1, 1, 1] = e_m @ e_m.conj().T

# The probability matrix encode \pi(0,0) = \pi(1,1) = 1/2
bb84_prob_mat = 1 / 2 * np.identity(2)

# %%
# ## The unentangled value of the BB84 extended nonlocal game
#
# It was shown in [@tomamichel2013monogamy] and [@johnston2016extended] that
#
# $$
# \omega(G_{BB84}) = \cos^2(\pi/8).
# $$
#
# This can be verified in `|toqito⟩` as follows.
#

# Calculate the unentangled value of the BB84 extended nonlocal game.
import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

# Define an ExtendedNonlocalGame object based on the BB84 game.
bb84 = ExtendedNonlocalGame(bb84_prob_mat, bb84_pred_mat)

# The unentangled value is cos(pi/8)**2 \approx 0.85356
print("The unentangled value is ", np.around(bb84.unentangled_value(), decimals=2))

# %%
# The BB84 game also exhibits strong parallel repetition. We can specify how many
# parallel repetitions for `|toqito⟩` to run. The example below provides an
# example of two parallel repetitions for the BB84 game.

# The unentangled value of BB84 under parallel repetition.
import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

# Define the bb84 game for two parallel repetitions.
bb84_2_reps = ExtendedNonlocalGame(bb84_prob_mat, bb84_pred_mat, 2)

# The unentangled value for two parallel repetitions is cos(pi/8)**4 \approx 0.72855
print("The unentangled value for two parallel repetitions is ", np.around(bb84_2_reps.unentangled_value(), decimals=2))

# %%
# It was shown in [@johnston2016extended] that the BB84 game possesses the property of strong
# parallel repetition. That is,
#
# $$
# \omega(G_{BB84}^r) = \omega(G_{BB84})^r
# $$
#
# for any integer $r$.
#
# ## The standard quantum value of the BB84 extended nonlocal game
#
# We can calculate lower bounds on the standard quantum value of the BB84 game
# using `|toqito⟩` as well.
#

# Calculate lower bounds on the standard quantum value of the BB84 extended nonlocal game.
import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

# Define an ExtendedNonlocalGame object based on the BB84 game.
bb84_lb = ExtendedNonlocalGame(bb84_prob_mat, bb84_pred_mat)

# The standard quantum value is cos(pi/8)**2 \approx 0.85356
print("The standard quantum value is ", np.around(bb84_lb.quantum_value_lower_bound(), decimals=2))

# %%
# From [@johnston2016extended], it is known that $\omega(G_{BB84}) =
# \omega^*(G_{BB84})$, however, if we did not know this beforehand, we could
# attempt to calculate upper bounds on the standard quantum value.
#
# There are a few methods to do this, but one easy way is to simply calculate the
# non-signaling value of the game as this provides a natural upper bound on the
# standard quantum value. Typically, this bound is not tight and usually not all
# that useful in providing tight upper bounds on the standard quantum value,
# however, in this case, it will prove to be useful.
#
# ## The non-signaling value of the BB84 extended nonlocal game
#
# Using `|toqito⟩`, we can see that $\omega_{ns}(G) = \cos^2(\pi/8)$.
#

# Calculate the non-signaling value of the BB84 extended nonlocal game.
import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

# Define an ExtendedNonlocalGame object based on the BB84 game.
bb84 = ExtendedNonlocalGame(bb84_prob_mat, bb84_pred_mat)

# The non-signaling value is cos(pi/8)**2 \approx 0.85356
print("The non-signaling value is ", np.around(bb84.nonsignaling_value(), decimals=2))

# %%
# So we have the relationship that
#
# $$
# \omega(G_{BB84}) = \omega^*(G_{BB84}) = \omega_{ns}(G_{BB84}) = \cos^2(\pi/8).
# $$
#
# It turns out that strong parallel repetition does *not* hold in the
# non-signaling scenario for the BB84 game. This was shown in [@russo2017extended], and we
# can observe this by the following snippet.
#

# The non-signaling value of BB84 under parallel repetition.
import numpy as np

mkdocs_gallery_thumbnail_path = 'figures/logo.svg'
from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

# Define the bb84 game for two parallel repetitions.
bb84_2_reps = ExtendedNonlocalGame(bb84_prob_mat, bb84_pred_mat, 2)

# The non-signaling value for two parallel repetitions is cos(pi/8)**4 \approx 0.73825
print(
    "The non-signaling value for two parallel repetitions is ", np.around(bb84_2_reps.nonsignaling_value(), decimals=2)
)

# %%
# Note that $0.73825 \geq \cos(\pi/8)^4 \approx 0.72855$ and therefore we
# have that
#
# $$
# \omega_{ns}(G^r_{BB84}) \not= \omega_{ns}(G_{BB84})^r
# $$
#
# for $r = 2$.
#
# Next, we will explore another well-known example, [The CHSH extended nonlocal game](../enlg_chsh), and see how its properties compare.
#
