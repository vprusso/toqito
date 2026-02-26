"""# Nonlocal games

In this tutorial, we are going to cover the notion of a *nonlocal game*; a
mathematical framework that abstractly models a physical system. The simplest
instance of a nonlocal game involves two players, Alice and Bob, who are not
allowed to communicate with each other once the game has started and who play
cooperatively against an adversary referred to as the referee.
"""
# %%
# A primary challenge that arises when studying these games is to determine the
# maximum probability with which Alice and Bob are able to achieve a winning
# outcome.
#
# This probability is highly dependent on the type of *strategy* that Alice and
# Bob use in the game. A *classical strategy* is one in which Alice and Bob have
# access to classical resources. The best that Alice and Bob can do using a
# classical strategy is known as the *classical value* of the game. Similarly, a
# *quantum strategy* is one in which Alice and Bob have access to quantum
# resources. The best that Alice and Bob can do using a quantum strategy is known
# as the *quantum value* of the game.
#
# Calculating the classical value of a game is NP-hard as we need to perform a
# brute-force check to see which strategy yields the classical value of the game.
#
# Using `|toqito⟩`, we will be able to specify a nonlocal game and be able to
# directly calculate the classical value and also place lower bounds on the
# quantum value.
#
# Further information beyond the scope of this tutorial on nonlocal games can be
# found in [@Cleve_2010_Consequences]. Further information on the lower bound technique can be found in
# [@Liang_2007_Bounds].
#
# ## Two-player nonlocal games
#
# A *two-player nonlocal game* consists of players that we give the names *Alice*
# and *Bob*:
#
# ![nonlocal game](../../../figures/alice_and_bob.svg){.center}
#
# <p style="text-align: center;"><em>The players: Alice and Bob.</em></p>
#
# Alice and Bob are in separate locations and cannot communicate once the game
# begins. Prior to the game however, Alice and Bob are free to communicate with
# each other. In addition to the players, there is also another party in this
# game that is referred to as the *referee*.
#
# ![nonlocal game](../../../figures/referee.svg){.center}
#
# <p style="text-align: center;"><em>The referee.</em></p>
#
# Alice and Bob want to play in a cooperative fashion against the referee.
#
# Now that we have set the stage with respect to the actors and actresses we will
# encounter in this game, let us see how the game is actually played.
#
# ![nonlocal game](../../../figures/nonlocal_game.svg){.center}
#
# <p style="text-align: center;"><em>A two-player nonlocal game.</em></p>
#
# A nonlocal game unfolds in the following manner.
#
# 1. The referee randomly generates questions denoted as $x$ and $y$.
#    The referee sends the question $x$ to Alice and the question
#    $y$ to Bob. The referee also keeps a copy of $x$ and
#    $y$ for reference.
#
# 2. Alice and Bob each receive their respective questions. They are then each
#    expected to respond to their questions with answers that we denote as
#    $a$ and $b$. Alice sends $a$ to the referee, and Bob
#    sends $b$.
#
# 3. When the referee receives $a$ and $b$ from Alice and Bob,
#    the referee evaluates a particular function that is predicated on the
#    questions $x$ and $y$ as well as the answers $a$ and
#    $b$. The outcome of this function is either $0$ or
#    $1$, where an outcome of $0$ indicates a loss for Alice and
#    Bob and an outcome of $1$ indicates a win for Alice and Bob.
#
#
# Alice and Bob's goal in the above game is to get the function in Step-3 to
# output a $1$, or equivalently, to indicate a winning outcome. This type
# of game is referred to as a *nonlocal game*.
#
# ## Classical and Quantum Strategies
#
# Now that we have the framework for a nonlocal game, we can consider the
# player's *strategy*; how the players play the game given access to certain
# resources. There are a number of strategies that the players can use, but for
# simplicity, we will restrict our attention to two types of strategies.
#
# 1. *Classical strategies*: The players answer the questions in a deterministic
#    manner.
#
# 2. *Quantum strategies*: The players make use of quantum resources in the form
#    of a shared quantum state and respective sets of measurements.
#
#
# ### Classical strategies
#
# A *classical strategy* for a nonlocal game is one where the players
# deterministically produce an output for every possible combination of inputs
# they may receive in the game. The *classical value* of a nonlocal game is the
# maximum probability achieved by the players over all classical strategies. For
# a nonlocal game, $G$, we use $\omega(G)$ to represent the classical
# value of $G$.
#
# One question you may have is whether a classical strategy can be improved by
# introducing randomness. If the players randomly select their answers, is it
# possible for them to do potentially better than if they had just played
# deterministically? As it happens, probabilistic classical strategies cannot
# perform any better than deterministic classical strategies.
#
# There is therefore no loss in generality in restricting our analysis of
# classical strategies to deterministic ones and it is assumed that when we use
# the term classical strategy that we implicitly mean a classical strategy that
# is played deterministically.
#
# ### Quantum strategies
#
# A *quantum strategy* for a nonlocal game is one where the players prepare a
# quantum state prior to the start of the game along with respective sets of
# measurements that they apply to their respective portions of the shared state
# during the game based on the questions they receive to generate their answers.
# The *quantum value* of a nonlocal game is the maximum probability achieved by
# the players over all quantum strategies. For a nonlocal game, $G$, we use
# $\omega^*(G)$ to represent the quantum value of $G$.
#
# ![nonlocal game quantum strategy](../../../figures/nonlocal_game_quantum_strategy.svg){.center}
# <p style="text-align: center;"> <em>A two-player nonlocal game invoking a quantum strategy.</em></p>
#
# Let us describe the high-level steps for how Alice and Bob play using a quantum
# strategy.
#
# 1. Alice and Bob prepare a state
#    $\sigma \in \text{D}(\mathcal{U} \otimes \mathcal{V})$ prior to the
#    start of the game. We use $\textsf{U}$ and $\textsf{V}$ to
#    denote the respective registers of spaces $\textsf{U}$ and
#    $\textsf{V}$.
#
# 2. The referee sends question $x \in \Sigma_A$ to Alice and
#    $y \in \Sigma_B$ to Bob.
#
# 3. Alice and Bob perform a *measurement* on their system. The outcome of
#    this measurement yields their answers $a \in \Gamma_A$ and
#    $b \in \Gamma_B$. Specifically, Alice and Bob have collections of
#    measurements
#
# $$
# \begin{aligned}
# \{ A_a^x : a \in \Gamma_{\text{A}} \} \subset \text{Pos}(\mathcal{U})
# \quad \text{and} \quad
# \{ B_b^y : b \in \Gamma_{\text{B}} \} \subset \text{Pos}(\mathcal{V}),
# \end{aligned}
# $$
#
#   such that the measurements satisfy
#
# $$
# \begin{aligned}
# \sum_{a \in \Gamma_A} A_a^x = \mathbb{I}_{\mathcal{U}}
# \quad \text{and} \quad
# \sum_{b \in \Gamma_B} B_b^y = \mathbb{I}_{\mathcal{V}}
# \end{aligned}
# $$
#
# *4.* The referee determines whether Alice and Bob win or lose, based on the
#    questions $x$ and $y$ as well as the answers $a$ and
#    $b$.
#
#
# For certain games, the probability that the players obtain a winning outcome is
# higher if they use a quantum strategy as opposed to a classical one. This
# striking separation is one primary motivation to study nonlocal games, as it
# provides examples of tasks that benefit from the manipulation of quantum
# information.
#
# ## Calculating the classical value
# (Coming soon)
#
# ## Calculating the quantum value
#
# The ability to calculate the quantum value for an arbitrary nonlocal game is a
# highly non-trivial task. Indeed, the quantum value is only known in special
# cases for certain nonlocal games.
#
# For an arbitrary nonlocal game, there exist approaches that place upper and
# lower bounds on the quantum value. The lower bound approach is calculated using
# the technique of semidefinite programming [@Liang_2007_Bounds]. While this method is efficient
# to carry out, it does not guarantee convergence to the quantum value (although
# in certain cases, it is attained).
#
# The primary idea of this approach is to note that fixing the measurements on one
# system yields the optimal measurements of the other system via an SDP. The
# algorithm proceeds in an iterative manner between two SDPs. In the first SDP, we
# assume that Bob's measurements are fixed, and Alice's measurements are to be
# optimized over. In the second SDP, we take Alice's optimized measurements from
# the first SDP and now optimize over Bob's measurements. This method is repeated
# until the quantum value reaches a desired numerical precision.
#
# For completeness, the first SDP where we fix Bob's measurements and optimize
# over Alice's measurements is given as SDP-1.
#
# $$
# \begin{equation}
# \begin{aligned}
# \textbf{SDP-1:} \quad & \\
# \text{maximize:} \quad & \sum_{(x,y \in \Sigma)} \pi(x,y)
# \sum_{(a,b) \in \Gamma}
# V(a,b|x,y)
# \langle B_b^y, A_a^x \rangle \\
# \text{subject to:} \quad & \sum_{a \in \Gamma_{\mathsf{A}}} =
# \tau, \qquad \qquad
# \forall x \in \Sigma_{\mathsf{A}}, \\
# \quad & A_a^x \in \text{Pos}(\mathcal{A}),
# \qquad
# \forall x \in \Sigma_{\mathsf{A}}, \
# \forall a \in \Gamma_{\mathsf{A}}, \\
# & \tau \in \text{D}(\mathcal{A}).
# \end{aligned}
# \end{equation}
# $$
#
# Similarly, the second SDP where we fix Alice's measurements and optimize over
# Bob's measurements is given as SDP-2.
#
# $$
# \begin{equation}
# \begin{aligned}
# \textbf{SDP-2:} \quad & \\
# \text{maximize:} \quad & \sum_{(x,y \in \Sigma)} \pi(x,y)
# \sum_{(a,b) \in \Gamma} V(a,b|x,y)
# \langle B_b^y, A_a^x \rangle \\
# \text{subject to:} \quad & \sum_{b \in \Gamma_{\mathsf{B}}} =
# \mathbb{I}, \qquad \qquad
# \forall y \in \Sigma_{\mathsf{B}}, \\
# \quad & B_b^y \in \text{Pos}(\mathcal{B}),
# \qquad \forall y \in \Sigma_{\mathsf{B}}, \
# \forall b \in \Gamma_{\mathsf{B}}.
# \end{aligned}
# \end{equation}
# $$
#
#
# ## Lower bounding the quantum value in `toqito`
#
# The `|toqito⟩` software implements both of these optimization problems using
# the `cvxpy` library. We see-saw between the two SDPs until the value we
# obtain reaches a specific precision threshold.
#
# As we are not guaranteed to obtain the true quantum value of a given nonlocal
# game as this approach can get stuck in a local minimum, the `|toqito⟩`
# function allows the user to specify an `iters` argument that runs the
# see-saw approach a number of times and then returns the highest of the values
# obtained.
#
# ### Example: Lower bounding the quantum value of the CHSH game
#
# Let us consider calculating the lower bound on the quantum value of the CHSH
# game.
#
# !!! note
#
#     As the CHSH game is a subtype of nonlocal game referred to as an XOR game,
#     we do not necessarily need to resort to this lower bound technique as there
#     exists a specific SDP formulation that one can use to directly compute the
#     quantum value of an XOR game. More information on how one defines the CHSH
#     game as well as this method to directly calculate the quantum value of an
#     XOR game is provided in [Calculating the Quantum and Classical Value of a Two-Player XOR Game](../xor_quantum_value)
#
# We will use the CHSH game here as an illustrative example as we already know
# what the optimal quantum value should be.
#
# The first step is to use `numpy` to encode a matrix that encapsulates the
# probabilities with which the questions are asked to Alice and Bob. As defined in
# the CHSH game, each of the four pairs
# $\{(0, 0), (0, 1), (1, 0), (1, 1)\}$ are all equally likely. We encode
# this in the matrix as follows.
#

# %%
# Creating the probability matrix.
import numpy as np

prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])

# %%
#  Next, we want to loop through all possible combinations of question and answer
# pairs and populate the $(a, b, x, y)^{th}$ entry of that matrix with a
# $1$ in the event that the winning condition is satisfied. Otherwise, if
# the winning condition is not satisfied for that particular choice of
# $a, b, x,$ and $y$, we place a $0$ at that position.
#
# The following code performs this operation and places the appropriate entries
# in this matrix into the `pred_mat` variable.


# Creating the predicate matrix.
import numpy as np

num_alice_inputs, num_alice_outputs = 2, 2
num_bob_inputs, num_bob_outputs = 2, 2

pred_mat = np.zeros((num_alice_outputs, num_bob_outputs, num_alice_inputs, num_bob_inputs))

for a_alice in range(num_alice_outputs):
    for b_bob in range(num_bob_outputs):
        for x_alice in range(num_alice_inputs):
            for y_bob in range(num_bob_inputs):
                if a_alice ^ b_bob == x_alice * y_bob:
                    pred_mat[a_alice, b_bob, x_alice, y_bob] = 1
pred_mat


# %%
# Now that we have both `prob_mat` and `pred_mat` defined, we can
# use `|toqito⟩` to determine the lower bound on the quantum value.

import numpy as np

from toqito.nonlocal_games.nonlocal_game import NonlocalGame

chsh = NonlocalGame(prob_mat, pred_mat)
# Multiple runs to avoid trap in suboptimal quantum value.
results = [np.around(chsh.quantum_value_lower_bound(), decimals=2) for _ in range(5)]
print(f"Maximum quantum value after multiple runs is: {max(results)}")

# %%
# In this case, we can see that the quantum value of the CHSH game is in fact
# attained as $\cos^2(\pi/8) \approx 0.85355$.
#
# ## The FFL game
#
# The *FFL (Fortnow, Feige, Lovasz) game* is a nonlocal game specified as
# follows.
#
# $$
# \begin{equation}
# \begin{aligned}
# &\pi(0, 0) = \frac{1}{3}, \quad
# \pi(0, 1) = \frac{1}{3}, \quad
# \pi(1, 0) = \frac{1}{3}, \quad
# \pi(1, 1) = 0, \\
# &(x,y) \in \Sigma_A \times \Sigma_B, \qquad \text{and} \qquad (a, b) \in \Gamma_A \times \Gamma_B,
# \end{aligned}
# \end{equation}
# $$
#
# where
#
# $$
# \begin{equation}
# \Sigma_A = \{0, 1\}, \quad \Sigma_B = \{0, 1\}, \quad \Gamma_A =
# \{0,1\}, \quad \text{and} \quad \Gamma_B = \{0, 1\}.
# \end{equation}
# $$
#
# Alice and Bob win the FFL game if and only if the following equation is
# satisfied
#
# $$
# \begin{equation}
# a \lor x = b \lor y.
# \end{equation}
# $$
#
# It is well-known that both the classical and quantum value of this nonlocal
# game is $2/3$ [@Cleve_2010_Consequences]. We can verify this fact using `|toqito⟩`.
# The following example encodes the FFL game. We then calculate the classical
# value and calculate lower bounds on the quantum value of the FFL game.
#
import numpy as np

from toqito.nonlocal_games.nonlocal_game import NonlocalGame

# Specify the number of inputs, and number of outputs.
num_alice_in, num_alice_out = 2, 2
num_bob_in, num_bob_out = 2, 2

# Define the probability matrix of the FFL game.
prob_mat = np.array([[1 / 3, 1 / 3], [1 / 3, 0]])


# Define the predicate matrix of the FFL game.
pred_mat = np.zeros((num_alice_out, num_bob_out, num_alice_in, num_bob_in))
for a_alice in range(num_alice_out):
    for b_bob in range(num_bob_out):
        for x_alice in range(num_alice_in):
            for y_bob in range(num_bob_in):
                if (a_alice or x_alice) != (b_bob or y_bob):
                    pred_mat[a_alice, b_bob, x_alice, y_bob] = 1
# Define the FFL game object.
ffl = NonlocalGame(prob_mat, pred_mat)
print(f"Classical value: {np.around(ffl.classical_value(), decimals=2)}")
print(f"Quantum value (lower bound): {np.around(ffl.quantum_value_lower_bound(), decimals=2)}")


# %%
# In this case, we obtained the correct quantum value of $2/3$, however,
# the lower bound technique is not guaranteed to converge to the true quantum
# value in general.
#
# ## Parallel repetitions of nonlocal games
#
# For classical strategies, it is known that parallel repetition does *not* hold
# for the CHSH game, that is:
#
# $$
# \begin{equation}
# w_c(\text{CHSH} \land \text{CHSH}) = 10/16 > 9/16 = w_c(\text{CHSH}) w_c(\text{CHSH}).
# \end{equation}
# $$
#
# ## Binary constraint system games
#
# The notion of a binary constraint system game was introduced in
# [@Cleve_2014_Characterization] and the following introductory material is
# extracted from that work.
#
# A *binary constraint system* (BCS) (sometimes also called a *linear system*
# (LCS)) consists of $n$ binary variables ``v_1``, ``v_2``, ..., ``v_n`` and
# $m$ constraints, ``c_1``, ``c_2``, ..., ``c_m``, where each ``c_j`` is
# a binary-valued function of a subset of the variables.
#
# A *binary constraint system game* (BCS game) is a two-player nonlocal game
# that is associated with a BCS. In a BCS game, the referee randomly selects a
# constraint ``c_s`` and one variable ``x_t`` from ``c_s``. The
# referee sends ``s`` to Alice and ``t`` to Bob. Alice returns a truth
# assignment to all variables in ``c_s`` and bob returns a truth assignment to
# variable ``x_t``. The verifier accepts the answer if and only if:
#
# 1. Alice's truth assignment satisfies the constraint ``c_s``;
# 2. Bob's truth assignment for ``x_t`` is consistent with Alice's.
#
#
# As an example, the CHSH game can be described as a BCS game:
#
# $$
# v_1 \oplus v_2 = 0 \quad \text{and} \quad v_1 \oplus v_2 = 1
# $$
#
# In `|toqito⟩`, we can encode this as a BCS game as follows
#
import numpy as np

from toqito.nonlocal_games import NonlocalGame

# mkdocs_gallery_thumbnail_path = 'figures/nonlocal_game.svg'
# Define constraints c_1 and c_2.
c_1 = np.zeros((2, 2))
c_2 = np.zeros((2, 2))

# Loop over variables and populate constraints.
for v_1 in range(2):
    for v_2 in range(2):
        if v_1 ^ v_2 == 0:
            c_1[v_1, v_2] = 1
        else:
            c_2[v_1, v_2] = 1

# Define the BCS game from the variables and constraints.
chsh_bcs = NonlocalGame.from_bcs_game([c_1, c_2])
# Classical value of CHSH is 3 / 4 = 0.75
print(f"Classical value: {np.around(chsh_bcs.classical_value(), decimals=2)}")
# Quantum value of CHSH is cos^2(pi/8) \approx 0.853
# Multiple runs to avoid trap in suboptimal quantum value.
results = [np.around(chsh_bcs.quantum_value_lower_bound(), decimals=2) for _ in range(5)]
print(f"Maximum quantum value after multiple runs is: {max(results)}")

# %%
