"""Extended nonlocal games
==========================

In this tutorial, we will define the concept of an *extended nonlocal game*.
Extended nonlocal games are a more general abstraction of nonlocal games
wherein the referee, who previously only provided questions and answers to the
players, now share a state with the players and is able to perform a
measurement on that shared state.
"""
# %%
# Every extended nonlocal game has a *value* associated to it. Analogously to
# nonlocal games, this value is a quantity that dictates how well the players can
# perform a task in the extended nonlocal game model when given access to certain
# resources. We will be using :code:`|toqito⟩` to calculate these quantities.

# %%
# We will also look at existing results in the literature on these values and be
# able to replicate them using :code:`|toqito⟩`. Much of the written content in
# this tutorial will be directly taken from :footcite:`Russo_2017_Extended`.

# %%
# Extended nonlocal games have a natural physical interpretation in the setting
# of tripartite steering :footcite:`Cavalcanti_2015_Detection` and in device-independent quantum scenarios
# :footcite:`Tomamichel_2013_AMonogamy`. For more information on extended nonlocal games, please refer to
# :footcite:`Johnston_2016_Extended` and :footcite:`Russo_2017_Extended`.

# %%
# The extended nonlocal game model
# --------------------------------
# An *extended nonlocal game* is similar to a nonlocal game in the sense that it
# is a cooperative game played between two players Alice and Bob against a
# referee. The game begins much like a nonlocal game, with the referee selecting
# and sending a pair of questions :math:`(x,y)` according to a fixed probability
# distribution. Once Alice and Bob receive :math:`x` and :math:`y`, they respond
# with respective answers :math:`a` and :math:`b`. Unlike a nonlocal game, the
# outcome of an extended nonlocal game is determined by measurements performed by
# the referee on its share of the state initially provided to it by Alice and
# Bob.
#
# .. figure:: ../../figures/extended_nonlocal_game.svg
#   :alt: extended nonlocal game
#   :align: center
#
#   An extended nonlocal game.
#
# Specifically, Alice and Bob's winning probability is determined by
# collections of measurements, :math:`V(a,b|x,y) \in \text{Pos}(\mathcal{R})`,
# where :math:`\mathcal{R} = \mathbb{C}^m` is a complex Euclidean space with
# :math:`m` denoting the dimension of the referee's quantum system--so if Alice
# and Bob's response :math:`(a,b)` to the question pair :math:`(x,y)` leaves the
# referee's system in the quantum state
#
# .. math::
#    \sigma_{a,b}^{x,y} \in \text{D}(\mathcal{R}),
#
# then their winning and losing probabilities are given by
#
# .. math::
#    \left\langle V(a,b|x,y), \sigma_{a,b}^{x,y} \right\rangle
#    \quad \text{and} \quad
#    \left\langle \mathbb{I} - V(a,b|x,y), \sigma_{a,b}^{x,y} \right\rangle.
#
#
# Strategies for extended nonlocal games
# ---------------------------------------
#
# An extended nonlocal game :math:`G` is defined by a pair :math:`(\pi, V)`,
# where :math:`\pi` is a probability distribution of the form
#
# .. math::
#    \pi : \Sigma_A \times \Sigma_B \rightarrow [0, 1]
#
# on the Cartesian product of two alphabets :math:`\Sigma_A` and
# :math:`\Sigma_B`, and :math:`V` is a function of the form
#
# .. math::
#    V : \Gamma_A \times \Gamma_B \times \Sigma_A \times \Sigma_B \rightarrow \text{Pos}(\mathcal{R})
#
# for :math:`\Sigma_A` and :math:`\Sigma_B` as above, :math:`\Gamma_A` and
# :math:`\Gamma_B` being alphabets, and :math:`\mathcal{R}` refers to the
# referee's space. Just as in the case for nonlocal games, we shall use the
# convention that
#
# .. math::
#    \Sigma = \Sigma_A \times \Sigma_B \quad \text{and} \quad \Gamma = \Gamma_A \times \Gamma_B
#
# to denote the respective sets of questions asked to Alice and Bob and the sets
# of answers sent from Alice and Bob to the referee.
#
# When analyzing a strategy for Alice and Bob, it may be convenient to define a
# function
#
# .. math::
#    K : \Gamma_A \times \Gamma_B \times \Sigma_A \times \Sigma_B \rightarrow \text{Pos}(\mathcal{R}).
#
# We can represent Alice and Bob's winning probability for an extended nonlocal
# game as
#
# .. math::
#    \sum_{(x,y) \in \Sigma} \pi(x,y) \sum_{(a,b) \in \Gamma} \left\langle V(a,b|x,y), K(a,b|x,y) \right\rangle.
#
# Standard quantum strategies for extended nonlocal games
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# A *standard quantum strategy* for an extended nonlocal game consists of
# finite-dimensional complex Euclidean spaces :math:`\mathcal{U}` for Alice and
# :math:`\mathcal{V}` for Bob, a quantum state :math:`\sigma \in
# \text{D}(\mathcal{U} \otimes \mathcal{R} \otimes \mathcal{V})`, and two
# collections of measurements
#
# .. math::
#    \{ A_a^x : a \in \Gamma_A \} \subset \text{Pos}(\mathcal{U})
#    \quad \text{and} \quad
#    \{ B_b^y : b \in \Gamma_B \} \subset \text{Pos}(\mathcal{V}),
#
# for each :math:`x \in \Sigma_A` and :math:`y \in \Sigma_B` respectively. As
# usual, the measurement operators satisfy the constraint that
#
# .. math::
#    \sum_{a \in \Gamma_A} A_a^x = \mathbb{I}_{\mathcal{U}}
#    \quad \text{and} \quad
#    \sum_{b \in \Gamma_B} B_b^y = \mathbb{I}_{\mathcal{V}},
#
# for each :math:`x \in \Sigma_A` and :math:`y \in \Sigma_B`.
#
# When the game is played, Alice and Bob present the referee with a quantum
# system so that the three parties share the state :math:`\sigma \in
# \text{D}(\mathcal{U} \otimes \mathcal{R} \otimes \mathcal{V})`. The referee
# selects questions :math:`(x,y) \in \Sigma` according to the distribution
# :math:`\pi` that is known to all participants in the game.
#
# The referee then sends :math:`x` to Alice and :math:`y` to Bob. At this point,
# Alice and Bob make measurements on their respective portions of the state
# :math:`\sigma` using their measurement operators to yield an outcome to send
# back to the referee. Specifically, Alice measures her portion of the state
# :math:`\sigma` with respect to her set of measurement operators :math:`\{A_a^x
# : a \in \Gamma_A\}`, and sends the result :math:`a \in \Gamma_A` of this
# measurement to the referee. Likewise, Bob measures his portion of the state
# :math:`\sigma` with respect to his measurement operators
# :math:`\{B_b^y : b \in \Gamma_B\}` to yield the outcome :math:`b \in \Gamma_B`,
# that is then sent back to the referee.
#
# At the end of the protocol, the referee measures its quantum system with
# respect to the measurement :math:`\{V(a,b|x,y), \mathbb{I}-V(a,b|x,y)\}`.
#
# The winning probability for such a strategy in this game :math:`G = (\pi,V)` is
# given by
#
# .. math::
#    \sum_{(x,y) \in \Sigma} \pi(x,y) \sum_{(a,b) \in \Gamma}
#    \left \langle A_a^x \otimes V(a,b|x,y) \otimes B_b^y,
#    \sigma
#    \right \rangle.
#
# For a given extended nonlocal game :math:`G = (\pi,V)`, we write
# :math:`\omega^*(G)` to denote the *standard quantum value* of :math:`G`, which
# is the supremum value of Alice and Bob's winning probability over all standard
# quantum strategies for :math:`G`.
#
# Unentangled strategies for extended nonlocal games
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# An *unentangled strategy* for an extended nonlocal game is simply a standard
# quantum strategy for which the state :math:`\sigma \in \text{D}(\mathcal{U}
# \otimes \mathcal{R} \otimes \mathcal{V})` initially prepared by Alice and Bob
# is fully separable.
#
# Any unentangled strategy is equivalent to a strategy where Alice and Bob store
# only classical information after the referee's quantum system has been provided
# to it.
#
# For a given extended nonlocal game :math:`G = (\pi, V)` we write
# :math:`\omega(G)` to denote the *unentangled value* of :math:`G`, which is the
# supremum value for Alice and Bob's winning probability in :math:`G` over all
# unentangled strategies. The unentangled value of any extended nonlocal game,
# :math:`G`, may be written as
#
# .. math::
#    \omega(G) = \max_{f, g}
#    \lVert
#    \sum_{(x,y) \in \Sigma} \pi(x,y)
#    V(f(x), g(y)|x, y)
#    \rVert
#
# where the maximum is over all functions :math:`f : \Sigma_A \rightarrow
# \Gamma_A` and :math:`g : \Sigma_B \rightarrow \Gamma_B`.
#
# Non-signaling strategies for extended nonlocal games
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# A *non-signaling strategy* for an extended nonlocal game consists of a function
#
# .. math::
#    K : \Gamma_A \times \Gamma_B \times \Sigma_A \times \Sigma_B \rightarrow \text{Pos}(\mathcal{R})
#
# such that
#
# .. math::
#    \sum_{a \in \Gamma_A} K(a,b|x,y) = \rho_b^y \quad \text{and} \quad \sum_{b \in \Gamma_B} K(a,b|x,y) = \sigma_a^x,
#
# for all :math:`x \in \Sigma_A` and :math:`y \in \Sigma_B` where
# :math:`\{\rho_b^y : y \in \Sigma_B, b \in \Gamma_B\}` and :math:`\{\sigma_a^x:
# x \in \Sigma_A, a \in \Gamma_A\}` are collections of operators satisfying
#
# .. math::
#    \sum_{a \in \Gamma_A} \sigma_a^x = \tau = \sum_{b \in \Gamma_B} \rho_b^y,
#
# for every choice of :math:`x \in \Sigma_A` and :math:`y \in \Sigma_B` and where
# :math:`\tau \in \text{D}(\mathcal{R})` is a density operator.
#
# For any extended nonlocal game, :math:`G = (\pi, V)`, the winning probability
# for a non-signaling strategy is given by
#
# .. math::
#    \sum_{(x,y) \in \Sigma} \pi(x,y) \sum_{(a,b) \in \Gamma} \left\langle V(a,b|x,y) K(a,b|x,y) \right\rangle.
#
# We denote the *non-signaling value* of :math:`G` as :math:`\omega_{ns}(G)`
# which is the supremum value of the winning probability of :math:`G` taken over
# all non-signaling strategies for Alice and Bob.
#
# Relationships between different strategies and values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For an extended nonlocal game, :math:`G`, the values have the following relationship:
#
#
# .. note::
#    .. math::
#        0 \leq \omega(G) \leq \omega^*(G) \leq \omega_{ns}(G) \leq 1.
#
#
# .. _ref-label-bb84_extended_nl_example:
#
# Example: The BB84 extended nonlocal game
# -----------------------------------------
#
# The *BB84 extended nonlocal game* is defined as follows. Let :math:`\Sigma_A =
# \Sigma_B = \Gamma_A = \Gamma_B = \{0,1\}`, define
#
# .. math::
#    \begin{equation}
#        \begin{aligned}
#            V(0,0|0,0) = \begin{pmatrix}
#                            1 & 0 \\
#                            0 & 0
#                         \end{pmatrix}, &\quad
#            V(1,1|0,0) = \begin{pmatrix}
#                            0 & 0 \\
#                            0 & 1
#                         \end{pmatrix}, \\
#            V(0,0|1,1) = \frac{1}{2}\begin{pmatrix}
#                            1 & 1 \\
#                            1 & 1
#                         \end{pmatrix}, &\quad
#            V(1,1|1,1) = \frac{1}{2}\begin{pmatrix}
#                            1 & -1 \\
#                            -1 & 1
#                         \end{pmatrix},
#        \end{aligned}
#    \end{equation}
#
# define
#
# .. math::
#    V(a,b|x,y) = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}
#
# for all :math:`a \not= b` or :math:`x \not= y`, define :math:`\pi(0,0) =
# \pi(1,1) = 1/2`, and define :math:`\pi(x,y) = 0` if :math:`x \not=y`.
#
# We can encode the BB84 game, :math:`G_{BB84} = (\pi, V)`, in :code:`numpy`
# arrays where :code:`prob_mat` corresponds to the probability distribution
# :math:`\pi` and where :code:`pred_mat` corresponds to the operator :math:`V`.

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_path = 'figures/extended_nonlocal_game.svg'
# sphinx_gallery_end_ignore
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
# The unentangled value of the BB84 extended nonlocal game
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# It was shown in :footcite:`Tomamichel_2013_AMonogamy` and :footcite:`Johnston_2016_Extended` that
#
# .. math::
#    \omega(G_{BB84}) = \cos^2(\pi/8).
#
# This can be verified in :code:`|toqito⟩` as follows.


# Calculate the unentangled value of the BB84 extended nonlocal game.
import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

# Define an ExtendedNonlocalGame object based on the BB84 game.
bb84 = ExtendedNonlocalGame(bb84_prob_mat, bb84_pred_mat)

# The unentangled value is cos(pi/8)**2 \approx 0.85356
print("The unentangled value is ", np.around(bb84.unentangled_value(), decimals=2))

# %%
# The BB84 game also exhibits strong parallel repetition. We can specify how many
# parallel repetitions for :code:`|toqito⟩` to run. The example below provides an
# example of two parallel repetitions for the BB84 game.

# The unentangled value of BB84 under parallel repetition.
import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

# Define the bb84 game for two parallel repetitions.
bb84_2_reps = ExtendedNonlocalGame(bb84_prob_mat, bb84_pred_mat, 2)

# The unentangled value for two parallel repetitions is cos(pi/8)**4 \approx 0.72855
print("The unentangled value for two parallel repetitions is ", np.around(bb84_2_reps.unentangled_value(), decimals=2))

# %%
# It was shown in :footcite:`Johnston_2016_Extended` that the BB84 game possesses the property of strong
# parallel repetition. That is,
#
# .. math::
#    \omega(G_{BB84}^r) = \omega(G_{BB84})^r
#
# for any integer :math:`r`.
#
# The standard quantum value of the BB84 extended nonlocal game
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can calculate lower bounds on the standard quantum value of the BB84 game
# using :code:`|toqito⟩` as well.

# Calculate lower bounds on the standard quantum value of the BB84 extended nonlocal game.
import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

# Define an ExtendedNonlocalGame object based on the BB84 game.
bb84_lb = ExtendedNonlocalGame(bb84_prob_mat, bb84_pred_mat)

# The standard quantum value is cos(pi/8)**2 \approx 0.85356
print("The standard quantum value is ", np.around(bb84_lb.quantum_value_lower_bound(), decimals=2))

# %%
# From :footcite:`Johnston_2016_Extended`, it is known that :math:`\omega(G_{BB84}) =
# \omega^*(G_{BB84})`, however, if we did not know this beforehand, we could
# attempt to calculate upper bounds on the standard quantum value.
#
# There are a few methods to do this, but one easy way is to simply calculate the
# non-signaling value of the game as this provides a natural upper bound on the
# standard quantum value. Typically, this bound is not tight and usually not all
# that useful in providing tight upper bounds on the standard quantum value,
# however, in this case, it will prove to be useful.
#
# The non-signaling value of the BB84 extended nonlocal game
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Using :code:`|toqito⟩`, we can see that :math:`\omega_{ns}(G) = \cos^2(\pi/8)`.

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
# .. math::
#    \omega(G_{BB84}) = \omega^*(G_{BB84}) = \omega_{ns}(G_{BB84}) = \cos^2(\pi/8).
#
# It turns out that strong parallel repetition does *not* hold in the
# non-signaling scenario for the BB84 game. This was shown in :footcite:`Russo_2017_Extended`, and we
# can observe this by the following snippet.

# The non-signaling value of BB84 under parallel repetition.
import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

# Define the bb84 game for two parallel repetitions.
bb84_2_reps = ExtendedNonlocalGame(bb84_prob_mat, bb84_pred_mat, 2)

# The non-signaling value for two parallel repetitions is cos(pi/8)**4 \approx 0.73825
print(
    "The non-signaling value for two parallel repetitions is ", np.around(bb84_2_reps.nonsignaling_value(), decimals=2)
)

# %%
# Note that :math:`0.73825 \geq \cos(\pi/8)^4 \approx 0.72855` and therefore we
# have that
#
# .. math::
#    \omega_{ns}(G^r_{BB84}) \not= \omega_{ns}(G_{BB84})^r
#
# for :math:`r = 2`.
#
# Example: The CHSH extended nonlocal game
# -----------------------------------------
#
# Let us now define another extended nonlocal game, :math:`G_{CHSH}`.
#
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
#
#
# Example: An extended nonlocal game with quantum advantage
# ----------------------------------------------------------
#
# So far, we have only seen examples of extended nonlocal games where the
# standard quantum and unentangled values are equal. Here we'll see an example of
# an extended nonlocal game where the standard quantum value is *strictly higher*
# than the unentangled value.
#
#
# Example: A monogamy-of-entanglement game with mutually unbiased bases
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Let :math:`\zeta = \exp(\frac{2 \pi i}{3})` and consider the following four
# mutually unbiased bases:
#
# .. math::
#    \begin{equation}\label{eq:MUB43}
#    \begin{aligned}
#      \mathcal{B}_0 &= \left\{ e_0,\: e_1,\: e_2 \right\}, \\
#      \mathcal{B}_1 &= \left\{ \frac{e_0 + e_1 + e_2}{\sqrt{3}},\:
#      \frac{e_0 + \zeta^2 e_1 + \zeta e_2}{\sqrt{3}},\:
#      \frac{e_0 + \zeta e_1 + \zeta^2 e_2}{\sqrt{3}} \right\}, \\
#      \mathcal{B}_2 &= \left\{ \frac{e_0 + e_1 + \zeta e_2}{\sqrt{3}},\:
#      \frac{e_0 + \zeta^2 e_1 + \zeta^2 e_2}{\sqrt{3}},\:
#      \frac{e_0 + \zeta e_1 + e_2}{\sqrt{3}} \right\}, \\
#      \mathcal{B}_3 &= \left\{ \frac{e_0 + e_1 + \zeta^2 e_2}{\sqrt{3}},\:
#      \frac{e_0 + \zeta^2 e_1 + e_2}{\sqrt{3}},\:
#      \frac{e_0 + \zeta e_1 + \zeta e_2}{\sqrt{3}} \right\}.
#    \end{aligned}
#    \end{equation}
#
# Define an extended nonlocal game :math:`G_{MUB} = (\pi,R)` so that
#
# .. math::
#
# 		\pi(0) = \pi(1) = \pi(2) = \pi(3) = \frac{1}{4}
#
# and :math:`R` is such that
#
# .. math::
# 		{ R(0|x), R(1|x), R(2|x) }
#
# represents a measurement with respect to the basis :math:`\mathcal{B}_x`, for
# each :math:`x \in \{0,1,2,3\}`.
#
# Taking the description of :math:`G_{MUB}`, we can encode this as follows.

# Define the monogamy-of-entanglement game defined by MUBs.
prob_mat = 1 / 4 * np.identity(4)

dim = 3
e_0, e_1, e_2 = basis(dim, 0), basis(dim, 1), basis(dim, 2)

eta = np.exp((2 * np.pi * 1j) / dim)
mub_0 = [e_0, e_1, e_2]
mub_1 = [
    (e_0 + e_1 + e_2) / np.sqrt(3),
    (e_0 + eta**2 * e_1 + eta * e_2) / np.sqrt(3),
    (e_0 + eta * e_1 + eta**2 * e_2) / np.sqrt(3),
]
mub_2 = [
    (e_0 + e_1 + eta * e_2) / np.sqrt(3),
    (e_0 + eta**2 * e_1 + eta**2 * e_2) / np.sqrt(3),
    (e_0 + eta * e_1 + e_2) / np.sqrt(3),
]
mub_3 = [
    (e_0 + e_1 + eta**2 * e_2) / np.sqrt(3),
    (e_0 + eta**2 * e_1 + e_2) / np.sqrt(3),
    (e_0 + eta * e_1 + eta * e_2) / np.sqrt(3),
]

# List of measurements defined from mutually unbiased basis.
mubs = [mub_0, mub_1, mub_2, mub_3]

num_in = 4
num_out = 3
pred_mat = np.zeros([dim, dim, num_out, num_out, num_in, num_in], dtype=complex)

pred_mat[:, :, 0, 0, 0, 0] = mubs[0][0] @ mubs[0][0].conj().T
pred_mat[:, :, 1, 1, 0, 0] = mubs[0][1] @ mubs[0][1].conj().T
pred_mat[:, :, 2, 2, 0, 0] = mubs[0][2] @ mubs[0][2].conj().T

pred_mat[:, :, 0, 0, 1, 1] = mubs[1][0] @ mubs[1][0].conj().T
pred_mat[:, :, 1, 1, 1, 1] = mubs[1][1] @ mubs[1][1].conj().T
pred_mat[:, :, 2, 2, 1, 1] = mubs[1][2] @ mubs[1][2].conj().T

pred_mat[:, :, 0, 0, 2, 2] = mubs[2][0] @ mubs[2][0].conj().T
pred_mat[:, :, 1, 1, 2, 2] = mubs[2][1] @ mubs[2][1].conj().T
pred_mat[:, :, 2, 2, 2, 2] = mubs[2][2] @ mubs[2][2].conj().T

pred_mat[:, :, 0, 0, 3, 3] = mubs[3][0] @ mubs[3][0].conj().T
pred_mat[:, :, 1, 1, 3, 3] = mubs[3][1] @ mubs[3][1].conj().T
pred_mat[:, :, 2, 2, 3, 3] = mubs[3][2] @ mubs[3][2].conj().T

# %%
# Now that we have encoded :math:`G_{MUB}`, we can calculate the unentangled value.

import numpy as np

g_mub = ExtendedNonlocalGame(prob_mat, pred_mat)
unent_val = g_mub.unentangled_value()
print("The unentangled value is ", np.around(unent_val, decimals=2))

# %%
# That is, we have that
#
# .. math::
#
#    \omega(G_{MUB}) = \frac{3 + \sqrt{5}}{8} \approx 0.65409.
#
# However, if we attempt to run a lower bound on the standard quantum value, we
# obtain.

import numpy as np

g_mub = ExtendedNonlocalGame(prob_mat, pred_mat)
q_val = g_mub.quantum_value_lower_bound()
print("The standard quantum value lower bound is ", np.around(q_val, decimals=2))

# %%
# Note that as we are calculating a lower bound, it is possible that a value this
# high will not be obtained, or in other words, the algorithm can get stuck in a
# local maximum that prevents it from finding the global maximum.
#
# It is uncertain what the optimal standard quantum strategy is for this game,
# but the value of such a strategy is bounded as follows
#
# .. math::
#
#    2/3 \geq \omega^*(G) \geq 0.6609.
#
# For further information on the :math:`G_{MUB}` game, consult :footcite:`Russo_2017_Extended`.


# %%
#
# Example: Modeling Bit Commitment Binding Failure
# ------------------------------------------------
#
# A bit commitment (BC) protocol is a cryptographic task involving two parties,
# Alice (the sender) and Bob (the receiver), which proceeds in two phases:
#
# 1.  **Commit Phase:** Alice chooses a secret bit :math:`b` and provides Bob with
#     a piece of evidence (in this case, a quantum state).
# 2.  **Reveal Phase:** At a later time, Alice announces the value of her bit, say
#     :math:`b'`, and provides information that allows Bob to use his evidence
#     from the commit phase to verify her claim.
#
# For the protocol to be secure, it must satisfy two fundamental properties:
#
# - **Hiding:** The evidence Bob receives during the **Commit Phase** must reveal
#   essentially no information about the value of Alice's bit :math:`b`. Bob
#   should not be able to distinguish the evidence for :math:`b=0` from the
#   evidence for :math:`b=1`.
# - **Binding:** Alice must be "locked in" to her choice after the **Commit Phase**.
#   She should not be able to change her mind and successfully convince Bob of a
#   different bit during the **Reveal Phase**. If she committed to :math:`b=0`,
#   she cannot successfully open the commitment as :math:`b=1`.
#
# The Mayers-Lo-Chau (MLC) no-go theorem :footcite:`Mayers_1997_Unconditionally,Lo_1997_Why`
# proves that no quantum protocol can be both perfectly hiding and binding. Here,
# we will use the :func:`.ExtendedNonlocalGame` framework not to prove the full
# theorem in its generality, but to illustrate the failure of
# the binding property. We will model a simplified, single-shot protocol to make
# the abstract threat of cheating concrete and quantifiable.
#
# The core of this impossibility proof lies in Alice's ability to use an
# Einstein-Podolsky-Rosen (EPR) type of attack :footcite:`Mayers_1997_Unconditionally,Lo_1997_Why`: she prepares an entangled state
# and shares one part with Bob, keeping the other. This entanglement allows her
# to delay her decision and "steer" the outcome to her advantage later on.
#
# The failure of binding property occurs when the protocol is *hiding* but not *binding*,
# allowing Alice to "change her mind." We can frame this as a game where Alice wins if she can
# successfully respond to a challenge from the referee (playing the role of Bob).
#
# Setting Up the Bit Commitment Game
#
# *   **Players:** The game models the two-party protocol between Alice (the
#     committer) and Bob (the receiver). To fit this cryptographic scenario into
#     our framework, we model the verifier, Bob, as the **Referee** who issues
#     the challenge. The 'player Bob' defined in the code is therefore a
#     necessary placeholder with trivial inputs and outputs, as his active role
#     is handled by the Referee.
#
# *   **The Challenge (Referee's Input):** The Referee (Bob) will challenge Alice
#     to reveal her commitment to either bit :math:`0` or bit :math:`1`. This is the Referee's
#     input, :math:`y`, which can be :math:`0` or :math:`1`. We assume he chooses between them with
#     equal probability, so :math:`\pi(y=0) = \pi(y=1) = 0.5`.
#
# *   **Alice's Strategy (The Quantum State):** In this game, Alice's entire
#     strategy is encapsulated in the initial quantum state she prepares and
#     shares with the Referee. Because she doesn't receive a question or return
#     an answer in the traditional sense, her inputs :math:`x` and outputs :math:`a` are trivial.
#
# *   **The Winning Condition (Referee's Measurement):** Alice wins if the state she
#     gives the Referee passes his verification test.
#
#     - If challenged with :math:`y=0`, the Referee measures with the projector for bit
#       :math:`0`, :math:`V(y=0) = |0\rangle\langle 0|`.
#     - If challenged with :math:`y=1`, the Referee measures with the projector for bit
#       :math:`1`, :math:`V(y=1) = |+\rangle\langle +|`.
#
# This choice of measurement bases is illustrative and inspired by states used in
# quantum key distribution. The power of the no-go theorem is that the protocol
# would remain insecure regardless of the specific orthogonal states Bob uses for
# his verification.
#
# Now, let's translate this game into code.

import numpy as np
from toqito.states import basis

# 1. Define Game Parameters
dim = 2
a_in, b_in = 1, 2
a_out, b_out = 1, 1

# 2. Define the Probability Matrix
bc_prob_mat = np.array([[0.5, 0.5]])

# 3. Define the Winning Condition Operators
e_0, e_1 = basis(2, 0), basis(2, 1)
e_p = (e_0 + e_1) / np.sqrt(2)

# Verification projector for bit 0 is a projection onto |0>.
proj_0 = e_0 @ e_0.conj().T
# Verification projector for bit 1 is a projection onto |+>.
proj_p = e_p @ e_p.conj().T

# 4. Assemble the Predicate Matrix V(a,b|x,y)
bc_pred_mat = np.zeros([dim, dim, a_out, b_out, a_in, b_in])

# If Referee's challenge is y=0 (b_in=0), the winning operator is proj_0.
bc_pred_mat[:, :, 0, 0, 0, 0] = proj_0

# If Referee's challenge is y=1 (b_in=1), the winning operator is proj_p.
bc_pred_mat[:, :, 0, 0, 0, 1] = proj_p

# %%
# Calculating Alice's Maximum Cheating Probability

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

bc_binding_game = ExtendedNonlocalGame(bc_prob_mat, bc_pred_mat)

# We use the NPA hierarchy (level 1) for a robust upper bound on the quantum value.
q_val = bc_binding_game.commuting_measurement_value_upper_bound(k=1)

print("Upper bound on the quantum value (Alice's cheating probability): ", np.around(q_val, decimals=5))

# %%
# Interpreting the Result
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The value returned by the solver, :math:`\approx 0.85355`, is not arbitrary. It represents
# the maximum possible success probability for Alice and can be derived from
# fundamental quantum mechanics.
#
# Alice's average winning probability is the expectation value of an operator
# representing the average of the two possible measurements:
#
# .. math::
#    P(\text{win}) = \mathbb{E}[V] = 0.5 \cdot \text{Tr}(V(y=0) \rho) + 0.5 \cdot \text{Tr}(V(y=1) \rho) = \text{Tr}\left( \left[0.5 \cdot (proj_0 + proj_p)\right] \rho \right)
#
# where :math:`\rho` is the state of the Referee's qubit. A key principle of
# quantum mechanics states that the maximum expectation value of an operator
# is its largest eigenvalue. The operator here is :math:`M = 0.5 \cdot (proj_0 + proj_p)`.
#
# The largest eigenvalue of this operator :math:`M` is:
#
# .. math::
#    \lambda_{\max}(M) = \frac{1}{2}\left(1 + \frac{1}{\sqrt{2}}\right) \approx 0.85355.
#
# We found this exact value using :code:`|toqito⟩`. In a secure protocol, the best
# Alice could hope for is a :math:`50`\%  success rate (by guessing the challenge). The
# fact that she can achieve over :math:`85`\% demonstrates a catastrophic failure of the
# *binding* property, confirming the no-go theorem.
#
# It is important to note that this value of :math:`\approx 0.85355` represents the maximum cheating
# probability for *this specific, imperfectly hiding game*. The full MLC no-go theorem
# makes an even stronger claim: for any protocol that is *perfectly hiding* (where Bob
# cannot gain any information at all about the bit before the reveal phase), Alice's
# cheating strategy can succeed with :math:`100`\% probability. This example demonstrates the
# fragility of the binding property, which worsens to a total failure in the
# perfectly hiding limit.

# %%
#
#
# References
# ----------
#
# .. footbibliography::
