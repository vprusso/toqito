"""# Post-Selection Games

Nonlocal games in which the referee may also discard a round. Hardy's
paradox is the motivating example: its post-selection game has classical
value $1/2$ and quantum value $1$. This tutorial computes the classical,
no-signaling and quantum bounds of post-selection games, their statistical
power, and the Bayes factors of a loophole-free Bell test analysed as a
CHSH game and as a Hardy game.
"""

# %%
# ## Overview
#
# In a nonlocal game the referee asks Alice and Bob questions $x, y$ with
# probability $\pi(x, y)$, receives answers $a, b$, and declares the round
# won with probability $V(a, b, x, y)$. A **post-selection game**
# [@callejarodriguez2026postselection] adds a third outcome: the referee
# keeps the round with probability $S(a, b, x, y)$ and discards it
# otherwise. The value of a behaviour $P(a, b | x, y)$ is the probability of
# winning a round given that it was kept,
#
# $$
# \omega_G(P) = \frac{\sum_{a, b, x, y} \pi(x, y) S(a, b, x, y) V(a, b, x, y) P(a, b | x, y)}
#                    {\sum_{a, b, x, y} \pi(x, y) S(a, b, x, y) P(a, b | x, y)},
# $$
#
# and the denominator $\gamma_G(P)$ is the probability that a round is kept.
#
# Hardy's paradox [@hardy1992quantum] is the statement that every local
# behaviour with $P(01|01) = P(10|10) = P(00|11) = 0$ also has
# $P(00|00) = 0$, while some quantum behaviours do not. As a post-selection
# game the referee keeps exactly these four events and the round is won on
# $(00|00)$. In the layout of [@callejarodriguez2026postselection], where the
# quadrants are the questions $(x, y)$ and the entries inside a quadrant are
# the answers $(a, b)$, the kept events and the winning event are
#
# $$
# S = \left(\begin{array}{cc|cc} 1 & & & 1 \\ & & & \\ \hline & & 1 & \\ 1 & & & \end{array}\right),
# \qquad
# V = \left(\begin{array}{cc|cc} 1 & & & \\ & & & \\ \hline & & & \\ & & & \end{array}\right).
# $$

# %%
import numpy as np

from toqito.nonlocal_games import NonlocalGame, PostSelectionGame

prob_mat = np.full((2, 2), 1 / 4)
select_mat = np.zeros((2, 2, 2, 2))
for a, b, x, y in [(0, 0, 0, 0), (0, 1, 0, 1), (1, 0, 1, 0), (0, 0, 1, 1)]:
    select_mat[a, b, x, y] = 1
pred_mat = np.zeros((2, 2, 2, 2))
pred_mat[0, 0, 0, 0] = 1

hardy = PostSelectionGame(prob_mat, pred_mat, select_mat)

# %%
# ## Classical, no-signaling and quantum values
#
# The value is a ratio of two linear functionals of the behaviour, so it is
# quasiconvex and its maximum over the local polytope is attained at a
# deterministic strategy. `classical_value` enumerates the deterministic
# strategies of one player and finds the other player's best response by
# Dinkelbach's method, whose subproblems separate over questions, so the
# cost grows with one player's strategy count rather than with the product
# of both.

# %%
value, (alice_outputs, bob_outputs) = hardy.classical_value(return_strategy=True)
print(f"Classical value: {value:.4f}")
print(f"Alice answers {alice_outputs} to her questions, Bob answers {bob_outputs}.")

# %%
# For the convex relaxations the value is maximized over the cone of
# unnormalized behaviours with the post-selection probability fixed to one,
# which turns the ratio into a linear objective. The no-signaling value is a
# linear program and the commuting-measurement value is bounded by the NPA
# hierarchy [@navascues2008convergent] in the same way as for nonlocal games.
# For Hardy's paradox both equal one: a PR box satisfies all three
# conditions with $P(00|00) = 1/2$, and so do suitable quantum behaviours.

# %%
print(f"No-signaling value: {hardy.nonsignaling_value():.4f}")
print(f"NPA upper bound at level 1+ab: {hardy.commuting_measurement_value_upper_bound(k='1+ab'):.4f}")

# %%
# ## Statistical power
#
# The gap between the quantum and classical values of the Hardy game is
# $1/2$, far larger than the CHSH game's $(\sqrt{2} - 1)/4 \approx 0.104$.
# Rounds are rare, though: the quantum value is only reached by behaviours
# whose post-selection probability is small. Playing $n$ rounds with a
# behaviour $P$, both the Bayes factor and the Chernoff bound on the
# $p$-value of the local hypothesis decay as $2^{-n W_G(P)}$, with
#
# $$
# W_G(P) = \gamma_G(P) \, D\big(\omega_G(P) \,\big\|\, \omega_\ell(G)\big),
# $$
#
# where $D$ is the binary relative entropy in bits and $\omega_\ell(G)$ the
# classical value. This is the statistical power of the behaviour.
#
# `post_selection_probability_upper_bound` maximizes the post-selection
# probability over NPA behaviours of at least a given value. At value one
# it recovers the largest Hardy probability, $(5\sqrt{5} - 11)/2$
# [@rabelo2012device], divided by the four question pairs.

# %%
kept, hardy_behaviour = hardy.post_selection_probability_upper_bound(1.0, k="1+ab", return_strategy=True)
print(f"Largest post-selection probability at value 1: {kept:.5f}")
print(f"(5 sqrt(5) - 11) / 8 = {(5 * np.sqrt(5) - 11) / 8:.5f}")
print(f"Statistical power of the Hardy behaviour: {hardy.statistical_power(hardy_behaviour):.4f}")

# %%
# A nonlocal game is a post-selection game that keeps every round. The CHSH
# game's ideal behaviour, read off the level-1 NPA optimum, has twice the
# statistical power of the ideal Hardy behaviour even though its gap is five
# times smaller.

# %%
chsh_pred = np.zeros((2, 2, 2, 2))
for a, b, x, y in np.ndindex(2, 2, 2, 2):
    chsh_pred[a, b, x, y] = float(a ^ b == x * y)
chsh = PostSelectionGame(prob_mat, chsh_pred, np.ones_like(chsh_pred))

_, chsh_behaviour = chsh.commuting_measurement_value_upper_bound(k=1, return_strategy=True)
print(f"CHSH classical value: {chsh.classical_value():.4f}")
print(f"Statistical power of the ideal CHSH behaviour: {chsh.statistical_power(chsh_behaviour):.4f}")

# %%
# ## Analysing a loophole-free Bell test
#
# The picture changes with noisy data. These are the raw counts of the
# loophole-free Bell test of [@shalm2015strong], relabelled by
# [@callejarodriguez2026postselection] into the layout above. Reading the data as a CHSH game counts every round; reading it as a
# Hardy game keeps about one round in fifteen thousand.

# %%
counts_block = np.array(
    [
        [6378, 3289, 6794, 2825],
        [3147, 44336240, 23230, 44311018],
        [6486, 21358, 106, 27562],
        [2818, 44302570, 30000, 44274530],
    ],
    dtype=float,
)
counts = np.zeros((2, 2, 2, 2))
for a, b, x, y in np.ndindex(2, 2, 2, 2):
    counts[a, b, x, y] = counts_block[2 * x + a, 2 * y + b]

rounds = counts.sum()
chsh_wins = (counts * chsh_pred).sum()
hardy_rounds = (counts * select_mat).sum()
hardy_wins = (counts * select_mat * pred_mat).sum()
print(f"Rounds: {rounds:.0f}; CHSH wins: {chsh_wins:.0f}")
print(f"Hardy rounds kept: {hardy_rounds:.0f}; Hardy wins: {hardy_wins:.0f}")

# %%
# With $k$ wins in $t$ kept rounds, the Bayes factor between the best local
# model, which wins each kept round with probability $\omega_\ell$, and a
# quantum model that wins with the observed frequency $\omega_Q = k / t$ is
#
# $$
# K = \exp_2\!\Big[k \log_2 \frac{\omega_\ell}{\omega_Q} + (t - k) \log_2 \frac{1 - \omega_\ell}{1 - \omega_Q}\Big].
# $$
#
# The CHSH analysis is inconclusive, while the Hardy analysis rejects local
# hidden variables decisively.


# %%
def bayes_factor(wins, kept_rounds, local_value):
    """Bayes factor of the local model against the observed winning frequency."""
    observed = wins / kept_rounds
    exponent = wins * np.log2(local_value / observed) + (kept_rounds - wins) * np.log2((1 - local_value) / (1 - observed))
    return 2.0**exponent


chsh_factor = bayes_factor(chsh_wins, rounds, chsh.classical_value())
hardy_factor = bayes_factor(hardy_wins, hardy_rounds, hardy.classical_value())
print(f"Bayes factor, CHSH game:  {chsh_factor:.4f}")
print(f"Bayes factor, Hardy game: {hardy_factor:.3e}")
print(f"Evidence per round, CHSH:  {-np.log2(chsh_factor) / rounds:.3e} bits")
print(f"Evidence per round, Hardy: {-np.log2(hardy_factor) / rounds:.3e} bits")

# %%
# The evidence per round is about sixteen times larger in the Hardy game, so
# the same conclusion would have needed sixteen times fewer rounds. The
# paper shows that this ratio is unbounded as the detection efficiency
# approaches $2/3$.
#
# ## A game without a paradox
#
# Post-selection games need not come from a possibilistic argument. Among
# all games on two questions and two answers with deterministic $S$ and $V$
# there are, up to relabelling, two that remain nontrivial near detection
# efficiency $2/3$: the Hardy game and the game below. Its no-signaling and
# quantum values are $2/3$, but only in the limit of vanishing
# post-selection probability, so the behaviour of largest statistical power
# sits strictly inside the quantum set.

# %%
select_neg = np.zeros((2, 2, 2, 2))
for a, b, x, y in [(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1)]:
    select_neg[a, b, x, y] = 1
pred_neg = np.zeros((2, 2, 2, 2))
pred_neg[0, 0, 0, 1] = pred_neg[0, 0, 1, 0] = 1
game_neg = PostSelectionGame(prob_mat, pred_neg, select_neg)

print(f"Classical value: {game_neg.classical_value():.4f}")
print(f"No-signaling value: {game_neg.nonsignaling_value():.4f}")
print(f"NPA upper bound at level 2: {game_neg.commuting_measurement_value_upper_bound(k=2):.4f}")

# %%
# Scanning the value and maximizing the post-selection probability at each
# step traces the statistical power along the boundary of the relaxation.
# It peaks near value $0.59$ at about $0.006$ bits per round, a quarter of
# the Hardy game's.

# %%
print(" value   kept    power")
for target in np.linspace(0.54, 0.64, 6):
    kept, behaviour = game_neg.post_selection_probability_upper_bound(target, k=2, return_strategy=True)
    print(f" {target:.2f}   {kept:.4f}  {game_neg.statistical_power(behaviour):.5f}")

# %%
# ## Generalized Hardy games
#
# Hardy's argument extends to $s$ questions and $k$ answers per player
# [@boschi1997ladder] [@meng2018hardy]: local behaviours with
# $P(a < b | 0, s-1) = 0$, $P(a < b | i, i-1) = 0$ and
# $P(a > b | i-1, i-1) = 0$ for $i = 1, \ldots, s-1$ also satisfy
# $P(a < b | s-1, s-1) = 0$. `PostSelectionGame.hardy` keeps exactly the
# events in these conditions, wins on the last one, and asks the $2s$
# question pairs involved uniformly. Every such game has classical value
# $1/2$ and quantum value $1$, and the statistical power of a behaviour of
# value one is its post-selection probability. The NPA bound on that
# probability matches the explicit qudit strategies of
# [@callejarodriguez2026postselection] for two answers, and leaves a gap of
# a few parts in a thousand for three.

# %%
print(" s  k  classical  NPA 1+ab  kept at value 1")
for num_inputs, num_outputs in [(2, 2), (3, 2), (4, 2), (2, 3)]:
    game = PostSelectionGame.hardy(num_inputs, num_outputs)
    classical = game.classical_value()
    quantum = game.commuting_measurement_value_upper_bound(k="1+ab")
    kept = game.post_selection_probability_upper_bound(1.0, k="1+ab")
    print(f" {num_inputs}  {num_outputs}  {classical:.4f}     {quantum:.4f}    {kept:.4f}")

# %%
# The same bounds for a nonlocal game come from `NonlocalGame`; the two
# classes agree whenever every round is kept.

# %%
chsh_nonlocal = NonlocalGame(prob_mat, chsh_pred)
print(f"NonlocalGame CHSH classical value: {chsh_nonlocal.classical_value():.4f}")
print(f"PostSelectionGame CHSH classical value: {chsh.classical_value():.4f}")
