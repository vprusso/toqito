"""Modeling Bit Commitment Binding Failure
=============================================

In this tutorial, we will model a quantum bit commitment protocol as an
extended nonlocal game where the "player" Alice attempts to cheat. Instead of
calculating a cooperative winning probability, we will quantify Alice's maximum
cheating probability. This allows us to provide a concrete illustration of the failure of the
*binding* property, a key aspect of the famous Mayers-Lo-Chau (MLC) no-go
theorem :footcite:`Mayers_1997_Unconditionally,Lo_1997_Why`.

"""

# %%
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
# we will use the :py:class:`~toqito.nonlocal_games.extended_nonlocal_game.ExtendedNonlocalGame` framework not to prove the full
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
