"""
The Pretty Good and Pretty Bad Measurements
===========================================

In this tutorial, we will explore the "pretty good measurement" (PGM) and its
novel counterpart, the "pretty bad measurement" (PBM), as introduced by
McIrvin et.al :footcite:`McIrvin_2024_Pretty`. These measurements
provide elegant, easy-to-construct tools for two opposing goals in quantum
information: state discrimination and state exclusion. PGM is useful for the former while PBM is of use for the latter.

We will verify their core properties and replicate some of the key numerical
results and figures from the paper using :code:`|toqito⟩`.
"""

# %%
# Background: Discrimination vs. Exclusion
# ----------------------------------------
#
# The standard quantum state discrimination task involves Alice sending Bob a
# quantum state :math:`\rho_i` chosen from a known ensemble
# :math:`\{(p_i, \rho_i)\}_{i=1}^k`. Bob's goal is to perform a measurement
# that maximizes his probability of correctly guessing the index :math:`i`.
# The best possible probability is denoted :math:`P_{\text{Best}}`. The "pretty good
# measurement" (PGM) is a well-known and effective heuristic for this task.
#
# The state exclusion task is the opposite: Bob wins if he correctly guesses
# a state that Alice *did not* send. This is equivalent to minimizing the
# probability of correctly guessing the state Alice *did* send. The minimum
# possible success probability for discrimination is :math:`P_{\text{Worst}}`. The
# "pretty bad measurement" (PBM) is a new heuristic designed to approximate
# this worst-case performance.
#
# A key result from :footcite:`McIrvin_2024_Pretty` is the tight relationship
# between the success probabilities of these two measurements:
#
# .. math::
#    P_{\text{PGM}} + (k-1)P_{\text{PBM}} = 1
#
# This implies a performance hierarchy against the optimal probabilities and the
# blind guessing probability (:math:`1/k`):
#
# .. math::
#    P_{\text{Best}} \ge P_{\text{PGM}} \ge \frac{1}{k} \ge P_{\text{PBM}} \ge P_{\text{Worst}}
#
# We will verify this hierarchy with a concrete example.

# %%
# Numerical Example: The Trine States
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Figure 3 of the paper :footcite:`McIrvin_2024_Pretty` analyzes the performance
# of these measurements for the three **trine states** with a uniform prior
# probability. The trine states are a classic example of a set that is
# antidistinguishable but not distinguishable.
#
# Our plan is to:
#
# 1.  Generate the trine states and assume uniform prior probabilities.
# 2.  Compute the optimal win/loss probabilities, :math:`P_{\text{Best}}`
#     and :math:`P_{\text{Worst}}`, using :code:`|toqito⟩`'s SDP solvers.
# 3.  Construct the PGM and PBM measurement operators.
# 4.  Calculate the success probabilities :math:`P_{\text{PGM}}` and :math:`P_{\text{PBM}}`.
# 5.  Print and compare all values, verifying the performance hierarchy and
#     the relationship between :math:`P_{\text{PGM}}` and :math:`P_{\text{PBM}}`.

import numpy as np
from toqito.measurements import pretty_bad_measurement, pretty_good_measurement
from toqito.states import trine
from toqito.state_opt import state_distinguishability, state_exclusion


def calculate_success_prob(
    states: list[np.ndarray],
    probs: list[float],
    povm_operators: list[np.ndarray],
) -> float:
    """Calculate the success probability Σ pᵢ Tr(ρᵢ Mᵢ).

    This helper is robust to `states` being either state vectors or density matrices.
    """
    success_prob = 0
    num_states = len(states)
    for i in range(num_states):
        state = states[i]
        op = povm_operators[i]
        # Check if input is a vector (pure state) or matrix (density matrix)
        if state.ndim == 1 or (state.ndim == 2 and min(state.shape) == 1):
            # It's a vector (or column/row vector)
            state_vec = state.flatten()
            prob_i = state_vec.conj().T @ op @ state_vec
        else:
            # It's a density matrix
            prob_i = np.trace(op @ state)
        success_prob += probs[i] * prob_i
    return np.real(success_prob)


# 1. Define the states and probabilities.
state_vectors = trine()
k = len(state_vectors)
probs = [1 / k] * k

print(f"Analyzing k={k} trine states with uniform probability.")

# 2. Compute the optimal benchmark values.
p_best, _ = state_distinguishability(state_vectors, probs)
p_worst, _ = state_exclusion(state_vectors, probs)

print(f"\nOptimal Benchmarks:")
print(f"  P_Best  = {p_best:.4f} (Max discrimination probability)")
print(f"  P_Worst = {p_worst:.4f} (Min discrimination probability)")

# 3. Compute the PGM and PBM operators.
pgm_operators = pretty_good_measurement(state_vectors, probs)
pbm_operators = pretty_bad_measurement(state_vectors, probs)

# 4. Calculate PGM and PBM success probabilities.
p_pgm = calculate_success_prob(state_vectors, probs, pgm_operators)
p_pbm = calculate_success_prob(state_vectors, probs, pbm_operators)

print(f"\nHeuristic Measurements:")
print(f"  P_PGM = {p_pgm:.4f}")
print(f"  P_PBM = {p_pbm:.4f}")

# 5. Verify the core relationship and the hierarchy.
relation_lhs = p_pgm + (k - 1) * p_pbm
print(f"\nVerifying P_PGM + (k-1)*P_PBM = 1:")
print(f"  {p_pgm:.4f} + ({k - 1})*{p_pbm:.4f} = {relation_lhs:.4f} -> {np.isclose(relation_lhs, 1)}")

print("\nVerifying hierarchy (P_Best >= P_PGM >= 1/k >= P_PBM >= P_Worst):")
print(f"  {p_best:.4f} >= {p_pgm:.4f}  ?  {p_best >= p_pgm or np.isclose(p_best, p_pgm)}")
print(f"  {p_pgm:.4f} >= {1 / k:.4f}  ?  {p_pgm >= 1 / k or np.isclose(p_pgm, 1 / k)}")
print(f"  {1 / k:.4f} >= {p_pbm:.4f}  ?  {1 / k >= p_pbm or np.isclose(1 / k, p_pbm)}")
print(f"  {p_pbm:.4f} >= {p_worst:.4f}?  {p_pbm >= p_worst or np.isclose(p_pbm, p_worst)}")


# %%
# The results perfectly match those presented in Figure 3 of the paper. For
# the trine states, the PGM is optimal (:math:`P_{\text{PGM}} = P_{\text{Best}} = 2/3`)
# and the worst possible measurement gives zero success probability
# (:math:`P_{\text{Worst}} = 0`).
#
# Our calculated value for the PBM is :math:`1/6 \approx 0.1667`, which is a
# good approximation of the true worst case. We also explicitly confirm
# the identity :math:`P_{\text{PGM}} + (k-1)P_{\text{PBM}} = 1` and the full
# performance hierarchy.

# %%
# Visualizing Performance on Random States
# ----------------------------------------
#
# Figures 4 and 5 in the paper :footcite:`McIrvin_2024_Pretty` show that for many randomly generated
# states, the PGM and PBM probabilities cluster around the blind guessing
# baseline of :math:`1/k`. We can reproduce a similar plot.
#
# We will generate 100 random ensembles of :math:`k=4` qubit states and plot
# the resulting :math:`P_{\text{PGM}}` and :math:`P_{\text{PBM}}` values.

import matplotlib.pyplot as plt

from toqito.rand import random_density_matrix

# Number of random ensembles to generate.
num_instances = 100
k = 4  # Number of states in each ensemble.
dim = 2  # Dimension of states (qubits).

pgm_results = []
pbm_results = []

for i in range(num_instances):
    # Generate a random ensemble of k density matrices.
    rand_states = [random_density_matrix(dim, seed=(i * k) + j) for j in range(k)]
    # Generate random prior probabilities.
    rand_probs = np.random.dirichlet(np.ones(k))

    # Calculate PGM and PBM probabilities.
    pgm_ops = pretty_good_measurement(rand_states, rand_probs)
    pbm_ops = pretty_bad_measurement(rand_states, rand_probs)

    pgm_prob = calculate_success_prob(rand_states, rand_probs, pgm_ops)
    pbm_prob = calculate_success_prob(rand_states, rand_probs, pbm_ops)
    pgm_results.append(pgm_prob)
    pbm_results.append(pbm_prob)

# Create the plot.
fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
sample_indices = range(num_instances)
ax.scatter(sample_indices, pgm_results, alpha=0.7, label="$P_{PGM}$", c="blue", s=20)
ax.scatter(sample_indices, pbm_results, alpha=0.7, label="$P_{PBM}$", c="red", s=20)

# Add blind guessing line for reference.
blind_guess_prob = 1 / k
ax.axhline(
    y=blind_guess_prob,
    color="black",
    linestyle="--",
    label=f"Blind Guessing (1/k = {blind_guess_prob:.2f})",
)

ax.set_xlabel("Random Instance Index", fontsize=12)
ax.set_ylabel("Discrimination Success Probability", fontsize=12)
ax.set_title(f"PGM and PBM Performance for {num_instances} Random Ensembles (k={k})", fontsize=14)
ax.legend()
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# %%
# This plot clearly illustrates the theoretical bounds. Every blue dot
# (:math:`P_{\text{PGM}}`) lies on or above the blind guessing line, and every
# red dot (:math:`P_{\text{PBM}}`) lies on or below it. This provides strong
# numerical evidence for the inequalities
# :math:`P_{\text{PGM}} \ge 1/k \ge P_{\text{PBM}}`, confirming that the PGM is
# always a better-than-random guess and the PBM is always a worse-than-random
# guess for state discrimination.
#
#
# References
# ----------
#
# .. footbibliography::
