"""# Separability and Entanglement Testing

Demonstrates how to use toqito to determine whether a quantum state
is separable or entangled. Covers product states, Bell states, PPT
entangled states, Werner states, and random density matrices.
"""

# %%
# ## Overview
#
# Determining whether a quantum state is separable or entangled is one
# of the most fundamental problems in quantum information theory. A
# bipartite state $\rho \in \text{D}(\mathcal{H}_A \otimes \mathcal{H}_B)$
# is *separable* if it can be written as a convex combination of product
# states:
#
# $$
# \rho = \sum_i p_i \, \rho_i^A \otimes \rho_i^B
# $$
#
# Otherwise, the state is *entangled*.
#
# !!! note
#
#     Determining separability is **NP-hard** in general
#     [@gurvits2002largest]. The `is_separable` function in `|toqito⟩`
#     applies a hierarchy of increasingly powerful tests, returning
#     `True` (separable) or `False` (entangled) based on which test
#     provides a definitive answer.
#
# ## The hierarchy of checks
#
# The `is_separable` function runs through 13 tests in order:
#
# 1. **Trivial cases** — one subsystem has dimension 1
# 2. **Pure states** — Schmidt rank check
# 3. **Separable ball** — closeness to maximally mixed state
# 4. **PPT criterion** — necessary condition; sufficient for 2×2 and 2×3
# 5. **Plücker coordinates** — 3×3 rank-4 PPT states
# 6. **Low-rank criteria** — Horodecki et al. (2000)
# 7. **Reduction criterion**
# 8. **Realignment / CCNR**
# 9. **Rank-1 perturbation of identity**
# 10. **2×N specific checks** — Johnston, Hildebrand
# 11. **Decomposable maps** — Ha-Kye, Breuer-Hall witnesses
# 12. **Symmetric extension hierarchy** — SDP-based (DPS)
#
# Let's see these in action.
#
# ## Example: Separable product state
#
# A product state $\rho = \rho_A \otimes \rho_B$ is always separable.

# %%
import numpy as np
from toqito.state_props import is_separable

# |+><+| ⊗ I/2
rho_a = np.array([[1, 1], [1, 1]]) / 2
rho_b = np.eye(2) / 2
rho_product = np.kron(rho_a, rho_b)

print(f"Product state is separable: {is_separable(rho_product)}")

# %%
# ## Example: Entangled Bell state
#
# The Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$
# is maximally entangled. The PPT criterion immediately detects this.

# %%
from toqito.states import bell

rho_bell = bell(0) @ bell(0).conj().T
print(f"Bell state is separable: {is_separable(rho_bell)}")

# %%
# ## Example: Werner states
#
# Werner states are a one-parameter family that interpolates between
# separable and entangled:
#
# $$
# W_p = p \, |\Phi^+\rangle\!\langle\Phi^+| + (1 - p) \frac{I}{4}
# $$
#
# For two qubits, a Werner state is separable if and only if
# $p \leq 1/3$ (by the PPT criterion).

# %%
from toqito.states import bell

phi_plus = bell(0) @ bell(0).conj().T
identity_4 = np.eye(4) / 4

# Separable Werner state (p = 0.2)
werner_sep = 0.2 * phi_plus + 0.8 * identity_4
print(f"Werner(p=0.2) is separable: {is_separable(werner_sep)}")

# Entangled Werner state (p = 0.5)
werner_ent = 0.5 * phi_plus + 0.5 * identity_4
print(f"Werner(p=0.5) is separable: {is_separable(werner_ent)}")

# %%
# ## Example: Random density matrices
#
# A random density matrix sampled from the Hilbert-Schmidt measure is
# generically entangled for dimensions larger than 2×2. We can verify
# this with `|toqito⟩`.

# %%
from toqito.rand import random_density_matrix

# Random 4x4 density matrix (2-qubit system)
rho_random = random_density_matrix(4, seed=42)
print(f"Random state is separable: {is_separable(rho_random)}")

# %%
# ## Example: States near the maximally mixed state
#
# The Gurvits-Barnum separable ball criterion guarantees that states
# sufficiently close to $I/d$ are separable. The maximally mixed state
# itself is trivially separable.

# %%
# Maximally mixed state
rho_mixed = np.eye(4) / 4
print(f"Maximally mixed state is separable: {is_separable(rho_mixed)}")

# Small perturbation of maximally mixed — still in the separable ball
rho_near_mixed = np.eye(4) / 4 + 0.001 * np.diag([1, -1, -1, 1])
rho_near_mixed = rho_near_mixed / np.trace(rho_near_mixed)
print(f"Near-mixed state is separable: {is_separable(rho_near_mixed)}")

# %%
# ## The symmetric extension hierarchy
#
# When the simpler criteria are inconclusive, `is_separable` falls back
# to the symmetric extension (DPS) hierarchy [@doherty2004complete].
# This is an SDP-based test: if a state has a $k$-symmetric extension
# for all $k$, it is separable. If it fails at any level $k$, it is
# entangled.
#
# The `level` parameter controls how many levels to check (default: 2).
# Higher levels are more powerful but computationally more expensive.
#
# $$
# \text{separable} \subset \cdots \subset k\text{-extendible}
# \subset \cdots \subset 2\text{-extendible} \subset \text{PPT}
# $$

# %%
# mkdocs_gallery_thumbnail_path = 'figures/quantum_state_distinguish.svg'
