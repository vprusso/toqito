"""
Antidistinguishability of Circulant States and the Eigenvalue Criterion
=======================================================================

In this tutorial, we investigate the antidistinguishability of a special
class of quantum states known as circulant states. We will numerically verify a
necessary and sufficient condition based on the eigenvalues of the
states' Gram matrix, as presented in the paper by Johnston et al.
:footcite:`Johnston_2025_Tight`.

This tutorial builds upon the concepts introduced in the
:ref:`sphx_glr_auto_examples_involvingstates_example_state_exclusion.py` tutorial.
"""

# %%
# Eigenvalue Criterion for Circulant States
# ---------------------------------------------
#
# A set of :math:`n` pure states is called *circulant* if its Gram matrix is
# circulant. A matrix is circulant if each of its rows is a cyclic shift of the
# row above it. Such sets of states have a high degree of symmetry and appear
# in various quantum information contexts.
#
# A key result from (Theorem 5.1) :footcite:`Johnston_2025_Tight` provides a
# simple and exact criterion for determining if a circulant set is
# antidistinguishable, based solely on the eigenvalues of its Gram matrix.
#
# The theorem states that a set of :math:`n` states with a circulant Gram
# matrix :math:`G` is **antidistinguishable if and only if** its eigenvalues
# :math:`\lambda_0 \ge \lambda_1 \ge \cdots \ge \lambda_{n-1}` satisfy the
# following inequality:
#
# .. math::
#    \sqrt{\lambda_0} \le \sum_{j=1}^{n-1} \sqrt{\lambda_j}
#
# This gives us a direct analytical test that is much more efficient than
# solving a full semidefinite program (SDP). We can use :code:`|toqito⟩` to
# verify this equivalence.

# %%
# Numerical Verification
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Our plan to verify this theorem is as follows:
#
# 1.  Generate a random circulant Gram matrix :math:`G` using
#     :func:`.random_circulant_gram_matrix`.
# 2.  Compute its eigenvalues and perform the **analytical check** using the
#     inequality from the theorem.
# 3.  Generate the corresponding set of state vectors from :math:`G` using
#     :func:`.vectors_from_gram_matrix`.
# 4.  Perform a **numerical check** by calling the high-level function
#     :func:`.is_antidistinguishable` to directly verify the property.
# 5.  Confirm that the analytical and numerical checks yield the same conclusion.

import numpy as np

from toqito.matrix_ops import vectors_from_gram_matrix
from toqito.rand import random_circulant_gram_matrix
from toqito.state_props import is_antidistinguishable

# 1. Define parameters and generate a random circulant Gram matrix.
n = 5
# Use a seed for reproducibility.
seed = 42

print(f"Generating a random {n}x{n} circulant Gram matrix (seed={seed})...")
gram_matrix = random_circulant_gram_matrix(n, seed=seed)

# 2. Perform the analytical check based on the eigenvalue criterion.
# Use 'eigvalsh' for Hermitian matrices; it's faster and returns real eigenvalues.
eigenvalues = np.linalg.eigvalsh(gram_matrix)
# Sort eigenvalues in descending order.
eigenvalues = np.sort(eigenvalues)[::-1]
lambda_0 = eigenvalues[0]
other_lambdas = eigenvalues[1:]

# The analytical check from the theorem:
lhs = np.sqrt(lambda_0)
# The sum of the square roots of the other eigenvalues.
# Use np.maximum to avoid numerical precision errors leading to sqrt of tiny negative numbers.
rhs = np.sum(np.sqrt(np.maximum(0, other_lambdas)))
analytical_is_ad = lhs <= rhs

print("\nANALYTICAL CHECK (from Theorem 5.1 of Johnston et al.):")
print(f"  sqrt(λ₀) = {lhs:.4f}")
print(f"  Σ sqrt(λⱼ) for j>0 = {rhs:.4f}")
print(f"  Is sqrt(λ₀) <= Σ sqrt(λⱼ)? {analytical_is_ad}")
print(f"  Conclusion: The set SHOULD BE antidistinguishable: {analytical_is_ad}")

# 3. Generate states from the Gram matrix for the numerical check.
states = vectors_from_gram_matrix(gram_matrix)

# 4. Perform the numerical check using |toqito⟩'s high-level function.
numerical_is_ad = is_antidistinguishable(states)

print("\nNUMERICAL CHECK (via is_antidistinguishable function):")
print(f"  Conclusion: The set IS antidistinguishable: {numerical_is_ad}")

# 5. Verify that both methods agree.
print("\n------------------------------------------------------")
print(f"Do the analytical and numerical results agree? {analytical_is_ad == numerical_is_ad}")
print("------------------------------------------------------")


# %%
# The results from both the analytical eigenvalue criterion and the numerical
# check using :code:`|toqito⟩`'s helper function agree, providing a concrete verification of Theorem 5.1 from
# :footcite:`Johnston_2025_Tight`. This demonstrates how a deep theoretical
# result can provide a powerful and efficient shortcut for a problem that would
# otherwise require a more computationally intensive optimization.
#
#
# References
# ----------
#
# .. footbibliography::
