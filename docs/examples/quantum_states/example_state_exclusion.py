"""Quantum state exclusion
===========================

In this tutorial, we are going to cover the problem of *quantum state
exclusion*. We are going to briefly describe the problem setting and then
describe how one may use :code:`|toqito⟩` to calculate the optimal probability
with which this problem can be solved for a number of different scenarios.
"""
# %%
# Quantum state exclusion is very closely related to the problem of quantum state
# distinguishability.
# It may be useful to consult the
# :ref:`sphx_glr_auto_examples_quantum_states_example_state_distinguishability.py`
# tutorial on this topic.
#
# Further information beyond the scope of this tutorial can be found in the text
# `:footcite:Pusey_2012_On` as well as the course :footcite:`Bandyopadhyay_2014_Conclusive`.
#
#
# The state exclusion problem
# ---------------------------
#
# The quantum state exclusion problem is phrased as follows.
#
# 1. Alice possesses an ensemble of :math:`n` quantum states:
#
#    .. math::
#        \begin{equation}
#            \eta = \left( (p_0, \rho_0), \ldots, (p_n, \rho_n)  \right),
#        \end{equation}
#
#    where :math:`p_i` is the probability with which state :math:`\rho_i` is
#    selected from the ensemble. Alice picks :math:`\rho_i` with probability
#    :math:`p_i` from her ensemble and sends :math:`\rho_i` to Bob.
#
# 2. Bob receives :math:`\rho_i`. Both Alice and Bob are aware of how the
#    ensemble is defined but he does *not* know what index :math:`i`
#    corresponding to the state :math:`\rho_i` he receives from Alice is.
#
# 3. Bob wants to guess which of the states from the ensemble he was *not* given.
#    In order to do so, he may measure :math:`\rho_i` to guess the index
#    :math:`i` for which the state in the ensemble corresponds.
#
# This setting is depicted in the following figure.
#
# .. figure:: ../../figures/quantum_state_distinguish.svg
#   :alt: quantum state exclusion
#   :align: center
#
#   The quantum state exclusion setting.
#
# .. note::
#    The primary difference between the quantum state distinguishability
#    scenario and the quantum state exclusion scenario is that in the former,
#    Bob want to guess which state he was given, and in the latter, Bob wants to
#    guess which state he was *not* given.
#
# Perfect state exclusion (antidistinguishability)
# ------------------------------------------------
#
# We say that if one is able to perfectly (without error) exclude all quantum states in a set, then the set of states is
# *antidistinguishable*.
#
# **Definition**: Let :math:`n` and :math:`d` be integers. A collection of quantum states
# :math:`S = \{|\psi_1\rangle, \ldots, |\psi_{n}\rangle\} \subset \mathbb{C}^d` are *antidistinguishable* if there exists
# a collection of positive operator value measurements :math:`\{M_1, \ldots, M_{n}\}` such that :math:`\langle \psi_i |
# M_i | \psi_i \rangle = 0` for all :math:`1 \leq i \leq n`.
#
# Recall that a collection of POVMs are positive semidefinite operators
# :math:`\{M_i : 1 \leq i \leq n\} \subset \mathbb{C}^d` that satisfy
#
# .. math::
#    \begin{equation}
#        \sum_{i=1}^{n} M_i = \mathbb{I}_{d}.
#    \end{equation}
#
# **Properties**:
#
# * If :math:`S` is distinguishable then it is antidistinguishable.
#
# * If :math:`n = 2` then :math:`S` is distinguishable if and only if :math:`S` is
#   antidistinguishable.
#
#   * Distinguishing one state from a pair of states is equivalent to excluding
#     one of the states from that pair.
#
# * If :math:`n \geq 3` then there are antidistinguishable sets that are not distinguishable.
#
#
# Example: Trine states
# ^^^^^^^^^^^^^^^^^^^^^
#
# The so-called *trine states* are a set of three states, each of dimension two defined as
#
# .. math::
#    \begin{equation}
#        |\psi_1\rangle = |0\rangle, \quad
#        |\psi_2\rangle = -\frac{1}{2}(|0\rangle + \sqrt{3}|1\rangle), \quad
#        |\psi_3\rangle = -\frac{1}{2}(|0\rangle - \sqrt{3}|1\rangle).
#    \end{equation}

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_path = 'figures/trine.png'
# sphinx_gallery_end_ignore
from toqito.states import trine

psi1, psi2, psi3 = trine()
print(f"|𝛙_1> = {psi1.reshape(1, -1)[0]}")
print(f"|𝛙_2> = {psi2.reshape(1, -1)[0]}")
print(f"|𝛙_3> = {psi3.reshape(1, -1)[0]}")

# %%
# The trine states are three states in two dimensions. So they can't be mutually orthogonal, but they are about "as close
# as you can get" for three states in two dimensions to be mutually orthogonal.
#
# .. figure:: ../../figures/trine.png
#   :alt: trine states
#   :align: center

from toqito.state_props import is_mutually_orthogonal
from toqito.states import trine

print(f"Are states mutually orthogonal: {is_mutually_orthogonal(trine())}")

# %%
# An interesting property of these states is that they are antidistinguishable but *not* distinguishable.

from toqito.state_props import is_antidistinguishable, is_distinguishable
from toqito.states import trine

print(f"Trine antidistinguishable: {is_antidistinguishable(trine())}")
print(f"Trine distinguishable: {is_distinguishable(trine())}")

# %%
# Here are a set of measurements that we can verify which satisfy the antidistinguishability constraints. We will see a
# method that we can use to obtain these directly later.
#
# .. math::
#    \begin{equation}
#        M_1 = \frac{2}{3} (\mathbb{I} - |\psi_1\rangle \langle \psi_1|), \quad
#        M_2 = \frac{2}{3} (\mathbb{I} - |\psi_2\rangle \langle \psi_2|), \quad
#        M_3 = \frac{2}{3} (\mathbb{I} - |\psi_3\rangle \langle \psi_3|).
#    \end{equation}

import numpy as np

M1 = 2 / 3 * (np.identity(2) - psi1 @ psi1.conj().T)
M2 = 2 / 3 * (np.identity(2) - psi2 @ psi2.conj().T)
M3 = 2 / 3 * (np.identity(2) - psi3 @ psi3.conj().T)

# %%
# In order for :math:`M_1`, :math:`M_2`, and :math:`M_3` to constitute as valid POVMs, each of these matrices must be
# positive semidefinite and we must ensure that :math:`\sum_{i \in \{1,2,3\}} M_i = \mathbb{I}_2`.

from toqito.matrix_props import is_positive_semidefinite

print(f"M_1 + M_2 + M_3 is identity: {np.allclose(M1 + M2 + M3, np.identity(2))}")
print(f"Is M_1 PSD: {is_positive_semidefinite(M1)}")
print(f"Is M_2 PSD: {is_positive_semidefinite(M2)}")
print(f"Is M_3 PSD: {is_positive_semidefinite(M3)}")

# %%
# Next, we must show that these measurements satisfy :math:`\langle \psi_i | M_i | \psi_i \rangle = 0`
# for all :math:`i \in \{1,2,3\}`.

print(f"<𝛙_1| M_1 |𝛙_1>: {np.around((psi1.reshape(1, -1)[0] @ M1 @ psi1)[0], decimals=5)}")
print(f"<𝛙_2| M_2 |𝛙_2>: {np.around((psi2.reshape(1, -1)[0] @ M2 @ psi2)[0], decimals=5)}")
print(f"<𝛙_3| M_3 |𝛙_3>: {np.around((psi3.reshape(1, -1)[0] @ M3 @ psi3)[0], decimals=5)}")

# %%
# Since we have exhibited a set of measurements :math:`\{M_i: i \in \{1,2,3\}\} \subset \text{Pos}(\mathbb{C^d})` that satisfy
#
#
# .. math::
#    \begin{equation}
#        \langle \psi_i | M_i | \psi_i \rangle = 0
#        \quad \text{and} \quad
#        \sum_{i \in \{1,2,3\}} M_i = \mathbb{I}_2
#    \end{equation}
#
# for all :math:`i`, we conclude that the trine states are antidistinguishable.
#
#
# An SDP for antidistinguishability
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Whether a collection of states :math:`\{|\psi_1 \rangle, |\psi_2\rangle, \ldots, |\psi_{n}\rangle \}` are antidistinguishable
# or not can be determined by the following semidefinite program (SDP).
#
# .. math::
#    \begin{equation}
#        \begin{aligned}
#            \text{minimize:} \quad & \sum_{i=1}^{n} \langle \psi_i | M_i | \psi_i \rangle  \\
#            \text{subject to:} \quad & \sum_{i=1}^{n} M_i = \mathbb{I}_{\mathcal{X}}, \\
#                                     & M_i \succeq 0 \quad \forall \ 1 \leq i \leq n.
#        \end{aligned}
#    \end{equation}
#
#
# Consider again the trine states from the previous example. We can determine that they are antidistinguishable by way of
# the antidistinguishability SDP.

from toqito.state_opt import state_exclusion
from toqito.states import trine

opt_value, measurements = state_exclusion(trine(), probs=[1, 1, 1], primal_dual="dual")
print(f"Optimal SDP value: {np.around(opt_value, decimals=2)}")

# %%
# The SDP not only gives us the optimal value, which is $0$ in this case, indicating that the states are
# antidistinguishable, but we also get a set of optimal measurement operators. These should look familiar to the
# measurements we explicitly constructed earlier.
#
#
# Antidistinguishability and :math:`(n-1)`-incoherence
# ----------------------------------------------------
#
# Antidistinguishability of a set of pure states is equivalent to a certain notion from the theory of quantum resources
# referred to as :math:`k`-incoherence :footcite:`Johnston_2022_Absolutely`:
#
# **Definition**: Let :math:`n` and :math:`k` be positive integers. Then :math:`X \in \text{Pos}(\mathbb{C} ^n)` is called
# :math:`k`-incoherent* if there exists a positive integer :math:`m`, a set
# :math:`S = \{|\psi_0\rangle, |\psi_1\rangle,\ldots, |\psi_{m-1}\rangle\} \subset \mathbb{C} ^n` with the property that
# each :math:`|\psi_i\rangle` has at most :math:`k` non-zero entries, and real scalars :math:`c_0, c_1, \ldots, c_{m-1} \geq 0`
# for which
#
# .. math::
#    X = \sum_{j=0}^{m-1} c_j |psi_j\rangle \langle \psi_j|.
#
# It turns out that antidistinguishability is equivalent to :math:`k`-incoherence in the :math:`k = n - 1` case.
# Reproducing one of the results from :footcite:`Johnston_2025_Tight`, we have the following theorem.
#
# **Theorem**: Let :math:`n \geq 2` be an integer and let :math:`S = \{|\phi_0\rangle, |\phi_1\rangle, \ldots, |\phi_{n-1}\rangle\}`.
# Then :math:`S` is antidistinguishable if and only if the Gram matrix :math:`G` is :math:`(n-1)`-incoherent.
#
# .. math::
#    G =
#    \begin{pmatrix}
#        1 & \langle \phi_0 | \phi_1 \rangle & \cdots & \langle \phi_0 | \phi_{n-1}\rangle \\
#        \langle \phi_1 | \phi_0 \rangle & 1 & \cdots & \langle \phi_1 | \phi_{n-1}\rangle \\
#        \vdots & \vdots & \ddots & \vdots \\
#        \langle \phi_{n-1} | \phi_0 \rangle & \langle \phi_{n-1} | \phi_1 \rangle & \cdots & 1
#    \end{pmatrix}
#
#
# As an example, we can generate a random collection of quantum states, obtain the corresponding Gram matrix, and compute
# whether the set of states are antidistinguishable and :math:`(n-1)`-incoherent.

from toqito.matrix_ops import vectors_to_gram_matrix
from toqito.matrix_props import is_k_incoherent
from toqito.rand import random_states
from toqito.state_props import is_antidistinguishable

n, d = 3, 3
states = random_states(n, d)
gram = vectors_to_gram_matrix(states)

print(f"Is Antidistinguishable: {is_antidistinguishable(states)}")
print(f"Is (n-1)-incoherent: {is_k_incoherent(gram, n - 1)}")

# %%
# As can be seen, whether the random set of states are antidistinguishable or not aligns with whether they are
# :math:`(n-1)`-incoherent or not as well.

# %%
#
#
# References
# ----------
#
# .. footbibliography::
