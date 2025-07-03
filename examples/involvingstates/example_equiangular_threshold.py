"""
Equiangular States and the Antidistinguishability Threshold
===========================================================

In this tutorial, we explore a sharp threshold for the antidistinguishability
of a special class of quantum states known as equiangular states. We will
numerically verify a tight bound presented in the paper "Tight bounds for
antidistinguishability and circulant sets of pure quantum states"
:footcite:`Johnston_2025_Tight` and visualize the "sharp cliff" where this
property changes.

This tutorial builds upon the concepts introduced in
:ref:`sphx_glr_auto_examples_involvingstates_example_state_exclusion.py`.
"""

# %%
# The Antidistinguishability Threshold for Equiangular States
# -----------------------------------------------------------
#
# A set of `n` pure states :math:`\{|\psi_0\rangle, \ldots, |\psi_{n-1}\rangle\}`
# is called *equiangular* if the absolute value of the inner product between
# any two distinct states is a constant, i.e.,
# :math:`|\langle \psi_i | \psi_j \rangle| = \gamma` for all :math:`i \neq j`.
#
# The paper by Johnston, Russo, and Sikora :footcite:`Johnston_2025_Tight`
# provides a simple and powerful necessary condition for a set of states to be
# antidistinguishable.
#
# **Corollary 4.2 from** :footcite:`Johnston_2025_Tight`: Let :math:`n \geq 2` be an
# integer and let :math:`S = \{|\psi_0\rangle, \ldots, |\psi_{n-1}\rangle\}`. If
#
# .. math::
#    |\langle \psi_i | \psi_j \rangle| > \frac{n-2}{n-1}
#    \quad \forall \ i \neq j,
#
# then :math:`S` is **not** antidistinguishable.
#
# Crucially, Example 3.3 in the paper demonstrates that this bound is *tight*.
# That is, a set of equiangular states with an inner product exactly equal
# to the threshold :math:`\gamma = \frac{n-2}{n-1}` *is* antidistinguishable.
# We can use :code:`|toqito⟩` to verify this sharp transition.

# %%
# Numerical Verification
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Let's test this for the case of :math:`n=4` states. The critical threshold
# for the inner product is :math:`\gamma_{\text{crit}} = (4-2)/(4-1) = 2/3`.
#
# We will construct two Gram matrices:
#
# #. One for a set of states with inner products *at* the threshold,
#    :math:`\gamma = 2/3`.
# #. Another for a set with inner products slightly *above* the threshold,
#    :math:`\gamma = 2/3 + \epsilon`.
#
# We expect the first set to be antidistinguishable and the second to not be.

import numpy as np
from toqito.matrix_ops import vectors_from_gram_matrix
from toqito.state_props import is_antidistinguishable

# Define parameters for n=4.
n = 4
gamma_crit = (n - 2) / (n - 1)
epsilon = 1e-5

# 1. Construct the Gram matrix AT the threshold.
gamma_at = gamma_crit
gram_at = (1 - gamma_at) * np.identity(n) + gamma_at * np.ones((n, n))
states_at = vectors_from_gram_matrix(gram_at)
is_ad_at = is_antidistinguishable(states_at)

print(f"For n={n}, the critical threshold is γ = {gamma_crit:.4f}")
print(f"Are states with γ = {gamma_at:.4f} antidistinguishable? {is_ad_at}")

# 2. Construct the Gram matrix slightly ABOVE the threshold.
gamma_above = gamma_crit + epsilon
gram_above = (1 - gamma_above) * np.identity(n) + gamma_above * np.ones((n, n))
states_above = vectors_from_gram_matrix(gram_above)
is_ad_above = is_antidistinguishable(states_above)

print(f"Are states with γ = {gamma_above:.4f} antidistinguishable? {is_ad_above}")


# %%
# Equivalence with (n-1)-Incoherence
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The core result of :footcite:`Johnston_2025_Tight` (Theorem 3.2) is that a
# set of `n` pure states is antidistinguishable if and only if its Gram matrix
# is :math:`(n-1)`-incoherent. The :code:`is_antidistinguishable` function
# uses a state exclusion SDP, while :code:`is_k_incoherent` uses a different
# SDP based on the structure of the Gram matrix itself. We can use this to
# cross-validate our results.

from toqito.matrix_props import is_k_incoherent

# Check for (n-1)-incoherence, which is k=3 for n=4.
k = n - 1

is_inc_at = is_k_incoherent(gram_at, k)
is_inc_above = is_k_incoherent(gram_above, k)

print(f"Is the Gram matrix with γ = {gamma_at:.4f} {k}-incoherent? {is_inc_at}")
print(f"Is the Gram matrix with γ = {gamma_above:.4f} {k}-incoherent? {is_inc_above}")
print(f"\nResults match: {is_ad_at == is_inc_at and is_ad_above == is_inc_above}")

# %%
# Visualizing the Threshold
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can make this "sharp cliff" even clearer by plotting the optimal value of
# the state exclusion SDP against the inner product :math:`\gamma`. The optimal
# value is 0 if and only if the states are antidistinguishable. The plot should
# show the value lifting off from 0 precisely at :math:`\gamma_{\text{crit}}`.

import matplotlib.pyplot as plt
from toqito.state_opt import state_exclusion
from toqito.matrix_props import is_positive_semidefinite

gamma_vals = np.linspace(0, 1, 101)
sdp_vals = []

for gamma in gamma_vals:
    # Construct the Gram matrix for this gamma.
    gram_matrix = (1 - gamma) * np.identity(n) + gamma * np.ones((n, n))

    # We can only generate real states if the Gram matrix is positive semidefinite.
    if is_positive_semidefinite(gram_matrix):
        states = vectors_from_gram_matrix(gram_matrix)
        # state_exclusion requires a list of probabilities; we assume uniform.
        probs = [1 / n] * n
        # The dual is often faster. The optimal value is what we care about.
        opt_val, _ = state_exclusion(states, probs, primal_dual="dual")
        sdp_vals.append(opt_val)
    else:
        # If not PSD, it's not a valid Gram matrix for a state ensemble.
        sdp_vals.append(np.nan)


fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
ax.plot(gamma_vals, sdp_vals, marker="o", markersize=4)
ax.axvline(
    x=gamma_crit,
    color="r",
    linestyle="--",
    label=f"Threshold $\\gamma = (n-2)/(n-1) \\approx {gamma_crit:.3f}$",
)
ax.set_xlabel("Inner Product $\\gamma = |\\langle \\psi_i | \\psi_j \\rangle|$", fontsize=12)
ax.set_ylabel("Optimal SDP Value (State Exclusion)", fontsize=12)
ax.set_title(f"Antidistinguishability Threshold for $n={n}$ Equiangular States", fontsize=14)
ax.legend(fontsize=12)
ax.grid(True)
plt.tight_layout()
plt.show()

# %%
# This plot, which numerically reproduces the results from Figure 2 of
# :footcite:`Johnston_2025_Tight`, shows that the optimal value of the state
# exclusion SDP is exactly 0 for all :math:`\gamma \leq (n-2)/(n-1)`,
# indicating that the states are perfectly antidistinguishable. The moment
# :math:`\gamma` exceeds this value, the SDP value becomes non-zero, meaning
# perfect exclusion is no longer possible.
#
#
# References
# ----------
#
# .. footbibliography::
