r"""# An extended nonlocal game with quantum advantage

In the previous tutorials on [The BB84 extended nonlocal game](../enlg_bb84) and [The CHSH extended nonlocal game](../enlg_chsh), we
saw examples where the standard quantum and unentangled values were equal
($\omega(G) = \omega^*(G)$). Here, we will construct an extended
nonlocal game where the standard quantum value is *strictly higher* than the
unentangled value, demonstrating a true quantum advantage.

"""
# %%
# ## A monogamy-of-entanglement game with mutually unbiased bases
#
# Let $\zeta = \exp(\frac{2 \pi i}{3})$ and consider the following four
# mutually unbiased bases:
#
# $$
#    \begin{equation}\label{eq:MUB43}
#    \begin{aligned}
#      \mathcal{B}_0 &= \left\{ e_0,\: e_1,\: e_2 \right\}, \\
#      \mathcal{B}_1 &= \left\{ \frac{e_0 + e_1 + e_2}{\sqrt{3}},\:
#      \frac{e_0 + \zeta^2 e_1 + \zeta e_2}{\sqrt{3}},\:
#      \frac{e_0 + \zeta e_1 + \zeta^2 e_2}{\sqrt{3}} \right\}, \\
#      \mathcal{B}_2 &= \left\{ \frac{e_0 + e_1 + \zeta e_2}{\sqrt{3}},\:
#      \frac{e_0 + \zeta^2 e_1 + \zeta^2 e_2}{\sqrt{3}},\:
#      \frac{e_0 + \zeta e_1 + e_2}{\sqrt{3}} \right\}, \\
#      \mathcal{B}_3 &= \left\{ \frac{e_0 + e_1 + \zeta^2 e_2}{\sqrt{3}},\:
#      \frac{e_0 + \zeta^2 e_1 + e_2}{\sqrt{3}},\:
#      \frac{e_0 + \zeta e_1 + \zeta e_2}{\sqrt{3}} \right\}.
#    \end{aligned}
#    \end{equation}
# $$
#
# Define an extended nonlocal game $G_{MUB} = (\pi,R)$ so that
#
# $$
# 		\pi(0) = \pi(1) = \pi(2) = \pi(3) = \frac{1}{4}
# $$
#
# and $R$ is such that
#
# $$
# 		\{ R(0|x), R(1|x), R(2|x) \}
# $$
#
# represents a measurement with respect to the basis $\mathcal{B}_x$, for
# each $x \in \{0,1,2,3\}$.
#
# Taking the description of $G_{MUB}$, we can encode this as follows.

import numpy as np

from toqito.states import basis

# The basis: {|0>, |1>}:
e_0, e_1 = basis(2, 0), basis(2, 1)

# Define the monogamy-of-entanglement game defined by MUBs.
prob_mat = 1 / 4 * np.identity(4)

dim = 3
e_0, e_1, e_2 = basis(dim, 0), basis(dim, 1), basis(dim, 2)

eta = np.exp((2 * np.pi * 1j) / dim)
mub_0 = [e_0, e_1, e_2]
mub_1 = [
    (e_0 + e_1 + e_2) / np.sqrt(3),
    (e_0 + eta**2 * e_1 + eta * e_2) / np.sqrt(3),
    (e_0 + eta * e_1 + eta**2 * e_2) / np.sqrt(3),
]
mub_2 = [
    (e_0 + e_1 + eta * e_2) / np.sqrt(3),
    (e_0 + eta**2 * e_1 + eta**2 * e_2) / np.sqrt(3),
    (e_0 + eta * e_1 + e_2) / np.sqrt(3),
]
mub_3 = [
    (e_0 + e_1 + eta**2 * e_2) / np.sqrt(3),
    (e_0 + eta**2 * e_1 + e_2) / np.sqrt(3),
    (e_0 + eta * e_1 + eta * e_2) / np.sqrt(3),
]

# List of measurements defined from mutually unbiased basis.
mubs = [mub_0, mub_1, mub_2, mub_3]

num_in = 4
num_out = 3
pred_mat = np.zeros([dim, dim, num_out, num_out, num_in, num_in], dtype=complex)

pred_mat[:, :, 0, 0, 0, 0] = mubs[0][0] @ mubs[0][0].conj().T
pred_mat[:, :, 1, 1, 0, 0] = mubs[0][1] @ mubs[0][1].conj().T
pred_mat[:, :, 2, 2, 0, 0] = mubs[0][2] @ mubs[0][2].conj().T

pred_mat[:, :, 0, 0, 1, 1] = mubs[1][0] @ mubs[1][0].conj().T
pred_mat[:, :, 1, 1, 1, 1] = mubs[1][1] @ mubs[1][1].conj().T
pred_mat[:, :, 2, 2, 1, 1] = mubs[1][2] @ mubs[1][2].conj().T

pred_mat[:, :, 0, 0, 2, 2] = mubs[2][0] @ mubs[2][0].conj().T
pred_mat[:, :, 1, 1, 2, 2] = mubs[2][1] @ mubs[2][1].conj().T
pred_mat[:, :, 2, 2, 2, 2] = mubs[2][2] @ mubs[2][2].conj().T

pred_mat[:, :, 0, 0, 3, 3] = mubs[3][0] @ mubs[3][0].conj().T
pred_mat[:, :, 1, 1, 3, 3] = mubs[3][1] @ mubs[3][1].conj().T
pred_mat[:, :, 2, 2, 3, 3] = mubs[3][2] @ mubs[3][2].conj().T

# %%
# Now that we have encoded $G_{MUB}$, we can calculate the unentangled value.

import numpy as np

from toqito.nonlocal_games.extended_nonlocal_game import ExtendedNonlocalGame

g_mub = ExtendedNonlocalGame(prob_mat, pred_mat)
unent_val = g_mub.unentangled_value()
print("The unentangled value is ", np.around(unent_val, decimals=2))

# %%
# That is, we have that
#
# $$
#    \omega(G_{MUB}) = \frac{3 + \sqrt{5}}{8} \approx 0.65409.
# $$
#
# However, if we attempt to run a lower bound on the standard quantum value, we
# obtain.

g_mub = ExtendedNonlocalGame(prob_mat, pred_mat)
q_val = g_mub.quantum_value_lower_bound()
print("The standard quantum value lower bound is ", np.around(q_val, decimals=2))
# mkdocs_gallery_thumbnail_path = 'figures/logo.png'

# %%
# Note that as we are calculating a lower bound, it is possible that a value this
# high will not be obtained, or in other words, the algorithm can get stuck in a
# local maximum that prevents it from finding the global maximum.
#
# It is uncertain what the optimal standard quantum strategy is for this game,
# but the value of such a strategy is bounded as follows
#
# $$
#    2/3 \geq \omega^*(G) \geq 0.6609.
# $$
#
# For further information on the $G_{MUB}$ game, consult [@Russo_2017_Extended].
