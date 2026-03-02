"""# Equiangular States and the Antidistinguishability Threshold

In this tutorial, we explore a sharp threshold for the antidistinguishability
of a special class of quantum states known as equiangular states. We will
numerically verify a tight bound presented in the paper by Johnson et.al
[@johnston2025tight] and visualize the "sharp cliff" where this
property changes.

This tutorial builds upon the concepts introduced in the [Quantum state exclusion](../state_exclusion) tutorial.
"""

# %%
# ## Antidistinguishability Threshold for Equiangular States
#
# A set of $n$ pure states $\{|\psi_0\rangle, \ldots, |\psi_{n-1}\rangle\}$
# is called *equiangular* if the absolute value of the inner product between
# any two distinct states is a constant, i.e.,
# $|\langle \psi_i | \psi_j \rangle| = \gamma$ for all $i \neq j$.
#
# Johnston et.al [@johnston2025tight]
# introduced a simple and powerful necessary condition for a set of states to be
# antidistinguishable.
#
# According to **Corollary 4.2 from** [@johnston2025tight], when $n \geq 2$, $S = \{|
# \psi_0\rangle, \ldots, |\psi_{n-1}\rangle\}$ is not anstidistinguishable if the following condition is satisfied.
#
# $$
# |\langle \psi_i | \psi_j \rangle| > \frac{n-2}{n-1}
# \quad \forall \ i \neq j,
# $$
#
#
#
# Crucially, Example 3.3 in the paper demonstrates that this bound is *tight*.
# That is, a set of equiangular states with an inner product exactly equal
# to the threshold $\gamma = \frac{n-2}{n-1}$ *is* antidistinguishable.
# We can use `|toqito⟩` to verify this sharp transition.
#
# ### Numerical Verification
#
# To demonstrate the tightness of this bound, we follow Example 3.3 from
# the paper [@johnston2025tight]. The Gram matrix for a set of $n$
# equiangular states is given by
#
# $$
# G = (1 - \gamma) I + \gamma J,
# $$
#
# where $I$ is the identity matrix and $J$ is the all-ones matrix.
#
# We will verify the threshold for the $n=4$ case, where the critical
# inner product is $\gamma_{\text{crit}} = (4-2)/(4-1) = 2/3$.
# Our verification plan is as follows:
#
# 1.  Construct the Gram matrix $G_{\text{at}}$ for $\gamma = \gamma_{\text{crit}}$.
# 2.  Construct a second Gram matrix $G_{\text{above}}$ for $\gamma$
# slightly greater than $\gamma_{\text{crit}}$.
# 3.  For each Gram matrix, use `|toqito⟩` to generate a corresponding set of
# state vectors.
# 4.  For each set of states, compute the **minimum probability of error** for state exclusion using
# the [`state_exclusion`][toqito.state_opt.state_exclusion.state_exclusion] function.
# 5.  Confirm that the states *at* the threshold are antidistinguishable
# (error probability is $0$) and the states *above* it are not (error probability is > $0$).
#
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
print(f"  - Optimal error probability is {opt_val_at:.2e}")
print(f"  - Is the set antidistinguishable? {is_ad_at} (as expected)")

# 2. Construct and test the Gram matrix slightly ABOVE the threshold.
gamma_above = gamma_crit + epsilon
gram_above = (1 - gamma_above) * np.identity(n) + gamma_above * np.ones((n, n))
states_above = vectors_from_gram_matrix(gram_above)
opt_val_above, _ = state_exclusion(states_above)
is_ad_above = np.isclose(opt_val_above, 0)

print(f"\nFor γ = {gamma_above:.4f} (above threshold):")
print(f"  - Optimal error probability is {opt_val_above:.2e}")
print(f"  - Is the set antidistinguishable? {is_ad_above} (as expected)")


# %%
# ## Antidistinguishability and (n-1)-Incoherence
#
# The core theoretical result of (Theorem 3.2) [@johnston2025tight]
# is that a set of $n$ pure states is antidistinguishable if and only if its
# Gram matrix is $(n-1)$-incoherent. Our numerical results above,
# obtained by solving the state exclusion SDP, implicitly verify this
# property for the Gram matrix as well.

# %%
# ### Visualizing the Threshold
#
# We can make this "sharp cliff" even clearer by plotting the optimal error
# probability of state exclusion against the inner product $\gamma$. To
# match the style of Figure 2 from [@johnston2025tight], we will
# plot this for several values of $n$.
#
# The value returned by [`state_exclusion`][toqito.state_opt.state_exclusion.state_exclusion] is the optimal
# probability of error. The plot should show this probability lifting off from
# $0$ precisely at the threshold $\gamma_{\text{crit}} = (n-2)/(n-1)$ for
# each respective $n$.
#
# A Gram matrix for equiangular states is positive semidefinite (and thus
# physically valid) if and only if $-1/(n-1) \leq \gamma \leq 1$.

import matplotlib.pyplot as plt

from toqito.matrix_props import is_positive_semidefinite

fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
gamma_range = np.linspace(0, 0.999, 101)
n_vals_to_plot = [2, 3, 4, 5, 10]

for n_val in n_vals_to_plot:
    error_probs = []
    gamma_crit_n = (n_val - 2) / (n_val - 1)

    for gamma in gamma_range:
        # The Gram matrix is only PSD in a specific range.
        if gamma < -1 / (n_val - 1):
            error_probs.append(np.nan)
            continue

        gram_matrix = (1 - gamma) * np.identity(n_val) + gamma * np.ones((n_val, n_val))

        if is_positive_semidefinite(gram_matrix):
            states = vectors_from_gram_matrix(gram_matrix)
            opt_val, _ = state_exclusion(states)
            # The returned optimal value is the error probability.
            error_probs.append(opt_val)
        else:
            # If not PSD, it's not a valid Gram matrix for a state ensemble.
            error_probs.append(np.nan)

    ax.plot(gamma_range, error_probs, label=f"$n={n_val}$ ($\\gamma_{{crit}} \\approx {gamma_crit_n:.2f}$)")


ax.set_xlabel("Inner Product $\\gamma = |\\langle \\psi_i | \\psi_j \\rangle|$", fontsize=12)
ax.set_ylabel("Optimal Exclusion Error Probability", fontsize=12)
ax.set_title("Antidistinguishability Threshold for Equiangular States", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True)
ax.set_ylim(bottom=-0.01)
plt.tight_layout()
plt.show()

# %%
# This plot, which numerically reproduces the results from Figure 2 of
# [@johnston2025tight], shows that the optimal
# probability of error is exactly $0$ for all
# $\gamma \leq (n-2)/(n-1)$, indicating that the states are perfectly
# antidistinguishable. The moment $\gamma$ exceeds this value, the
# probability becomes non-zero, meaning perfect exclusion is no longer possible.
