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
# A set of :math:`n` pure states :math:`\{|\psi_0\rangle, \ldots, |\psi_{n-1}\rangle\}`
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
# We will construct Gram matrices for states with inner products *at* the
# threshold and slightly *above* it. To robustly check for
# antidistinguishability, we will directly use the :func:`~toqito.state_opt.state_exclusion`
# function. A set of states is antidistinguishable if and only if the optimal
# value of the state exclusion SDP is 0.

import numpy as np
from toqito.matrix_ops import vectors_from_gram_matrix
from toqito.state_opt import state_exclusion

# Define parameters for n=4.
n = 4
gamma_crit = (n - 2) / (n - 1)
# Use a larger epsilon to make the effect numerically obvious.
epsilon = 0.01

print(f"For n={n}, the critical threshold is γ = {gamma_crit:.4f}")

# 1. Construct and test the Gram matrix AT the threshold.
gamma_at = gamma_crit
gram_at = (1 - gamma_at) * np.identity(n) + gamma_at * np.ones((n, n))
states_at = vectors_from_gram_matrix(gram_at)
opt_val_at, _ = state_exclusion(states_at)
is_ad_at = np.isclose(opt_val_at, 0)

print(f"\nFor γ = {gamma_at:.4f} (at threshold):")
print(f"  - Optimal SDP value is {opt_val_at:.2e}")
print(f"  - Is the set antidistinguishable? {is_ad_at} (as expected)")

# 2. Construct and test the Gram matrix slightly ABOVE the threshold.
gamma_above = gamma_crit + epsilon
gram_above = (1 - gamma_above) * np.identity(n) + gamma_above * np.ones((n, n))
states_above = vectors_from_gram_matrix(gram_above)
opt_val_above, _ = state_exclusion(states_above)
is_ad_above = np.isclose(opt_val_above, 0)

print(f"\nFor γ = {gamma_above:.4f} (above threshold):")
print(f"  - Optimal SDP value is {opt_val_above:.2e}")
print(f"  - Is the set antidistinguishable? {is_ad_above} (as expected)")


# %%
# Antidistinguishability and (n-1)-Incoherence
# ---------------------------------------------
#
# The core theoretical result of :footcite:`Johnston_2025_Tight` (Theorem 3.2)
# is that a set of :math:`n` pure states is antidistinguishable if and only if its
# Gram matrix is :math:`(n-1)`-incoherent. Our numerical results above,
# obtained by solving the state exclusion SDP, implicitly verify this
# property for the Gram matrix as well.

# %%
# Visualizing the Threshold
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can make this "sharp cliff" even clearer by plotting the optimal value of
# the state exclusion SDP against the inner product :math:`\gamma`. The optimal
# value is 0 if and only if the states are antidistinguishable. The plot should
# show the value lifting off from 0 precisely at :math:`\gamma_{\text{crit}}`.
#
# A Gram matrix for equiangular states is positive semidefinite (and thus
# physically valid) if and only if :math:`-1/(n-1) \leq \gamma \leq 1`. Our
# plot covers the most interesting part of this range.

import matplotlib.pyplot as plt
from toqito.matrix_props import is_positive_semidefinite

gamma_vals = np.linspace(0, 0.8, 101)
sdp_vals = []

for gamma in gamma_vals:
    # Construct the Gram matrix for this gamma.
    gram_matrix = (1 - gamma) * np.identity(n) + gamma * np.ones((n, n))

    # We can only generate states if the Gram matrix is positive semidefinite.
    if is_positive_semidefinite(gram_matrix):
        states = vectors_from_gram_matrix(gram_matrix)
        # state_exclusion can be called without probabilities for a uniform ensemble.
        opt_val, _ = state_exclusion(states)
        sdp_vals.append(opt_val)
    else:
        # If not PSD, it's not a valid Gram matrix for a state ensemble.
        sdp_vals.append(np.nan)


fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
ax.plot(gamma_vals, sdp_vals, marker=".", linestyle="-", markersize=5)
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
# :footcite:`Johnston_2025_Tight` for the :math:`n=4` case, shows that the optimal
# value of the state exclusion SDP is exactly :math:`0` for all
# :math:`\gamma \leq (n-2)/(n-1)`, indicating that the states are perfectly
# antidistinguishable. The moment :math:`\gamma` exceeds this value, the SDP
# value becomes non-zero, meaning perfect exclusion is no longer possible.
#
#
# References
# ----------
#
# .. footbibliography::
