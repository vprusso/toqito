"""# Extended nonlocal games

This tutorial defines the extended nonlocal game model, where the referee shares
a quantum state with Alice and Bob and performs measurements on that state to
determine the game outcome. Covers standard quantum, unentangled, and
non-signaling strategies as well as the relationships between their values.
"""
# %%
# Every extended nonlocal game has a *value* associated to it. Analogously to
# nonlocal games, this value is a quantity that dictates how well the players can
# perform a task in the extended nonlocal game model when given access to certain
# resources. We will be using `|toqito⟩` to calculate these quantities.

# %%
# We will also look at existing results in the literature on these values and be
# able to replicate them using `|toqito⟩`. Much of the written content in
# this tutorial will be directly taken from [@russo2017extended].

# %%
# Extended nonlocal games have a natural physical interpretation in the setting
# of tripartite steering [@cavalcanti2015detection] and in device-independent quantum scenarios
# [@tomamichel2013monogamy]. For more information on extended nonlocal games, please refer to
# [@johnston2016extended] and [@russo2017extended].

# %%
# ## The extended nonlocal game model
# An *extended nonlocal game* is similar to a nonlocal game in the sense that it
# is a cooperative game played between two players Alice and Bob against a
# referee. The game begins much like a nonlocal game, with the referee selecting
# and sending a pair of questions $(x,y)$ according to a fixed probability
# distribution. Once Alice and Bob receive $x$ and $y$, they respond
# with respective answers $a$ and $b$. Unlike a nonlocal game, the
# outcome of an extended nonlocal game is determined by measurements performed by
# the referee on its share of the state initially provided to it by Alice and
# Bob.
#
# ![extended nonlocal game](../../../figures/extended_nonlocal_game.svg){.center}
# <p style="text-align: center;"><em>An extended nonlocal game. </em></p>
#
# Specifically, Alice and Bob's winning probability is determined by
# collections of measurements, $V(a,b|x,y) \in \text{Pos}(\mathcal{R})$,
# where $\mathcal{R} = \mathbb{C}^m$ is a complex Euclidean space with
# $m$ denoting the dimension of the referee's quantum system--so if Alice
# and Bob's response $(a,b)$ to the question pair $(x,y)$ leaves the
# referee's system in the quantum state
#
# $$
# \sigma_{a,b}^{x,y} \in \text{D}(\mathcal{R}),
# $$
#
# then their winning and losing probabilities are given by
#
# $$
# \left\langle V(a,b|x,y), \sigma_{a,b}^{x,y} \right\rangle
# \quad \text{and} \quad
# \left\langle \mathbb{I} - V(a,b|x,y), \sigma_{a,b}^{x,y} \right\rangle.
# $$
#
#
# ## Strategies for extended nonlocal games
#
# An extended nonlocal game $G$ is defined by a pair $(\pi, V)$,
# where $\pi$ is a probability distribution of the form
#
# $$
# \pi : \Sigma_A \times \Sigma_B \rightarrow [0, 1]
# $$
#
# on the Cartesian product of two alphabets $\Sigma_A$ and
# $\Sigma_B$, and $V$ is a function of the form
#
# $$
# V : \Gamma_A \times \Gamma_B \times \Sigma_A \times \Sigma_B \rightarrow \text{Pos}(\mathcal{R})
# $$
#
# for $\Sigma_A$ and $\Sigma_B$ as above, $\Gamma_A$ and
# $\Gamma_B$ being alphabets, and $\mathcal{R}$ refers to the
# referee's space. Just as in the case for nonlocal games, we shall use the
# convention that
#
# $$
# \Sigma = \Sigma_A \times \Sigma_B \quad \text{and} \quad \Gamma = \Gamma_A \times \Gamma_B
# $$
#
# to denote the respective sets of questions asked to Alice and Bob and the sets
# of answers sent from Alice and Bob to the referee.
#
# When analyzing a strategy for Alice and Bob, it may be convenient to define a
# function
#
# $$
# K : \Gamma_A \times \Gamma_B \times \Sigma_A \times \Sigma_B \rightarrow \text{Pos}(\mathcal{R}).
# $$
#
# We can represent Alice and Bob's winning probability for an extended nonlocal
# game as
#
# $$
# \sum_{(x,y) \in \Sigma} \pi(x,y) \sum_{(a,b) \in \Gamma} \left\langle V(a,b|x,y), K(a,b|x,y) \right\rangle.
# $$
#
# ### Standard quantum strategies for extended nonlocal games
#
# A *standard quantum strategy* for an extended nonlocal game consists of
# finite-dimensional complex Euclidean spaces $\mathcal{U}$ for Alice and
# $\mathcal{V}$ for Bob, a quantum state $\sigma \in
# \text{D}(\mathcal{U} \otimes \mathcal{R} \otimes \mathcal{V})$, and two
# collections of measurements
#
# $$
# \{ A_a^x : a \in \Gamma_A \} \subset \text{Pos}(\mathcal{U})
# \quad \text{and} \quad
# \{ B_b^y : b \in \Gamma_B \} \subset \text{Pos}(\mathcal{V}),
# $$
#
# for each $x \in \Sigma_A$ and $y \in \Sigma_B$ respectively. As
# usual, the measurement operators satisfy the constraint that
#
# $$
# \sum_{a \in \Gamma_A} A_a^x = \mathbb{I}_{\mathcal{U}}
# \quad \text{and} \quad
# \sum_{b \in \Gamma_B} B_b^y = \mathbb{I}_{\mathcal{V}},
# $$
#
# for each $x \in \Sigma_A$ and $y \in \Sigma_B$.
#
# When the game is played, Alice and Bob present the referee with a quantum
# system so that the three parties share the state $\sigma \in
# \text{D}(\mathcal{U} \otimes \mathcal{R} \otimes \mathcal{V})$. The referee
# selects questions $(x,y) \in \Sigma$ according to the distribution
# $\pi$ that is known to all participants in the game.
#
# The referee then sends $x$ to Alice and $y$ to Bob. At this point,
# Alice and Bob make measurements on their respective portions of the state
# $\sigma$ using their measurement operators to yield an outcome to send
# back to the referee. Specifically, Alice measures her portion of the state
# $\sigma$ with respect to her set of measurement operators $\{A_a^x
# : a \in \Gamma_A\}$, and sends the result $a \in \Gamma_A$ of this
# measurement to the referee. Likewise, Bob measures his portion of the state
# $\sigma$ with respect to his measurement operators
# $\{B_b^y : b \in \Gamma_B\}$ to yield the outcome $b \in \Gamma_B$,
# that is then sent back to the referee.
#
# At the end of the protocol, the referee measures its quantum system with
# respect to the measurement $\{V(a,b|x,y), \mathbb{I}-V(a,b|x,y)\}$.
#
# The winning probability for such a strategy in this game $G = (\pi,V)$ is
# given by
#
# $$
# \sum_{(x,y) \in \Sigma} \pi(x,y) \sum_{(a,b) \in \Gamma}
# \left \langle A_a^x \otimes V(a,b|x,y) \otimes B_b^y,
# \sigma
# \right \rangle.
# $$
#
# For a given extended nonlocal game $G = (\pi,V)$, we write
# $\omega^*(G)$ to denote the *standard quantum value* of $G$, which
# is the supremum value of Alice and Bob's winning probability over all standard
# quantum strategies for $G$.
#
# ### Unentangled strategies for extended nonlocal games
#
# An *unentangled strategy* for an extended nonlocal game is simply a standard
# quantum strategy for which the state $\sigma \in \text{D}(\mathcal{U}
# \otimes \mathcal{R} \otimes \mathcal{V})$ initially prepared by Alice and Bob
# is fully separable.
#
# Any unentangled strategy is equivalent to a strategy where Alice and Bob store
# only classical information after the referee's quantum system has been provided
# to it.
#
# For a given extended nonlocal game $G = (\pi, V)$ we write
# $\omega(G)$ to denote the *unentangled value* of $G$, which is the
# supremum value for Alice and Bob's winning probability in $G$ over all
# unentangled strategies. The unentangled value of any extended nonlocal game,
# $G$, may be written as
#
# $$
# \omega(G) = \max_{f, g}
# \lVert
# \sum_{(x,y) \in \Sigma} \pi(x,y)
# V(f(x), g(y)|x, y)
# \rVert
# $$
#
# where the maximum is over all functions $f : \Sigma_A \rightarrow
# \Gamma_A$ and $g : \Sigma_B \rightarrow \Gamma_B$.
#
# ### Non-signaling strategies for extended nonlocal games
#
# A *non-signaling strategy* for an extended nonlocal game consists of a function
#
# $$
# K : \Gamma_A \times \Gamma_B \times \Sigma_A \times \Sigma_B \rightarrow \text{Pos}(\mathcal{R})
# $$
#
# such that
#
# $$
# \sum_{a \in \Gamma_A} K(a,b|x,y) = \rho_b^y \quad \text{and} \quad \sum_{b \in \Gamma_B} K(a,b|x,y) = \sigma_a^x,
# $$
#
# for all $x \in \Sigma_A$ and $y \in \Sigma_B$ where
# $\{\rho_b^y : y \in \Sigma_B, b \in \Gamma_B\}$ and $\{\sigma_a^x:
# x \in \Sigma_A, a \in \Gamma_A\}$ are collections of operators satisfying
#
# $$
# \sum_{a \in \Gamma_A} \sigma_a^x = \tau = \sum_{b \in \Gamma_B} \rho_b^y,
# $$
#
# for every choice of $x \in \Sigma_A$ and $y \in \Sigma_B$ and where
# $\tau \in \text{D}(\mathcal{R})$ is a density operator.
#
# For any extended nonlocal game, $G = (\pi, V)$, the winning probability
# for a non-signaling strategy is given by
#
# $$
# \sum_{(x,y) \in \Sigma} \pi(x,y) \sum_{(a,b) \in \Gamma} \left\langle V(a,b|x,y) K(a,b|x,y) \right\rangle.
# $$
#
# We denote the *non-signaling value* of $G$ as $\omega_{ns}(G)$
# which is the supremum value of the winning probability of $G$ taken over
# all non-signaling strategies for Alice and Bob.
#
# ### Relationships between different strategies and values
#
# For an extended nonlocal game, $G$, the values have the following relationship:
#
#
# !!! note
#
#     $$
#     0 \leq \omega(G) \leq \omega^*(G) \leq \omega_{ns}(G) \leq 1.
#     $$
#
# Now that we have established the theoretical framework for extended nonlocal
# games, we can explore some concrete examples. In the following tutorials, we
# will construct well-known games such as the BB84 and CHSH extended nonlocal
# games and use `|toqito⟩` to calculate their various values.
#
# %%
mkdocs_gallery_thumbnail_path = 'figures/extended_nonlocal_game.svg'
# We will start by examining the BB84 extended nonlocal game in [The BB84 extended nonlocal game](../enlg_bb84)
