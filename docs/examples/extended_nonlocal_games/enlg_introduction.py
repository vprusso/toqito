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

# %%
# Now that we have established the theoretical framework for extended nonlocal
# games, we can explore some concrete examples. In the following tutorials, we
# will construct well-known games such as the BB84 and CHSH extended nonlocal
# games and use :code:`|toqito⟩` to calculate their various values.
#
# We will start by examining the BB84 extended nonlocal game in :ref:`sphx_glr_auto_examples_extended_nonlocal_games_enlg_bb84.py`
#
# References
# ----------
#
# .. footbibliography::
