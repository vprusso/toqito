"""Two-player post-selection game."""

import cvxpy
import numpy as np
from scipy.special import xlogy

from toqito._sdp import solve_sdp
from toqito.nonlocal_games._common import Assemblage, nonsignaling_cone, validate_game
from toqito.nonlocal_games.nonlocal_game import _STRATEGY_CHUNK_BYTES, QuantumStrategy, _deterministic_strategies
from toqito.state_opt.npa_hierarchy import npa_constraints

# Dinkelbach's method stops once no deterministic strategy beats the current ratio by more than this.
_RATIO_TOL = 1e-13


class PostSelectionGame:
    r"""Create a two-player post-selection game object.

    A *post-selection game* [@callejarodriguez2026postselection] is a nonlocal game in which a round can be won,
    lost, or discarded by the referee. Alice and Bob receive questions \(x, y\) with probability \(\pi(x, y)\) and
    answer \(a, b\). The referee keeps the round with probability \(S(a, b, x, y)\) and, if it is kept, declares it won
    with probability \(V(a, b, x, y)\). The value of a behaviour \(P(a, b | x, y)\) is the probability of winning a
    round given that it was kept,

    \[
        \omega_G(P) = \frac{\sum_{a, b, x, y} \pi(x, y) S(a, b, x, y) V(a, b, x, y) P(a, b | x, y)}
                           {\sum_{a, b, x, y} \pi(x, y) S(a, b, x, y) P(a, b | x, y)},
    \]

    and the denominator is the probability \(\gamma_G(P)\) that a round is kept. The value is undefined for
    behaviours that never pass post-selection. A nonlocal game is the special case \(S \equiv 1\).

    Post-selection games formalize possibilistic proofs of nonlocality such as Hardy's paradox [@hardy1992quantum],
    see [`PostSelectionGame.hardy`][toqito.nonlocal_games.post_selection_game.PostSelectionGame.hardy], and their
    statistical power can exceed that of any nonlocal game at low detection efficiency.

    The value is a ratio of two linear functionals of the behaviour, so it is quasiconvex, and the classical,
    non-signaling and commuting-measurement values are all computed by the methods used for nonlocal games, with
    the normalization of the behaviour replaced by a normalization of the post-selection probability.

    Examples:
        Hardy's paradox as a post-selection game. Its classical value is \(1/2\) and its quantum value is \(1\),
        attained only by behaviours whose post-selection probability is at most
        \((5\sqrt{5} - 11)/8 \approx 0.0225\).

        ```python exec="1" source="above" result="text"
        from toqito.nonlocal_games import PostSelectionGame
        game = PostSelectionGame.hardy()
        print(f"{game.classical_value():.4f}")
        print(f"{game.commuting_measurement_value_upper_bound(k='1+ab'):.4f}")
        print(f"{game.post_selection_probability_upper_bound(1.0, k='1+ab'):.4f}")
        ```

    """

    def __init__(self, prob_mat: np.ndarray, pred_mat: np.ndarray, select_mat: np.ndarray) -> None:
        r"""Construct post-selection game object.

        Args:
            prob_mat: A matrix whose (x, y)-entry gives the probability that the referee asks Alice `x` and Bob `y`.
            pred_mat: A four-dimensional matrix whose (a, b, x, y)-entry gives the probability \(V(a, b, x, y)\) that
                a kept round with answers `a`, `b` to questions `x`, `y` is won.
            select_mat: A four-dimensional matrix of the same shape whose (a, b, x, y)-entry gives the probability
                \(S(a, b, x, y)\) that such a round is kept.

        Raises:
            ValueError: If the shapes do not match, if `prob_mat` is not a probability distribution, if `pred_mat` or
                `select_mat` has an entry outside \([0, 1]\), or if no answer event is ever kept.

        """
        prob_mat = np.asarray(prob_mat)
        pred_mat = np.asarray(pred_mat)
        select_mat = np.asarray(select_mat)
        validate_game(prob_mat, pred_mat, pred_ndim=4)
        if select_mat.shape != pred_mat.shape:
            raise ValueError(
                f"Invalid: `select_mat` has shape {select_mat.shape} but `pred_mat` has shape {pred_mat.shape}."
            )
        for name, mat in (("pred_mat", pred_mat), ("select_mat", select_mat)):
            if np.any(np.real(mat) < -1e-12) or np.any(np.real(mat) > 1 + 1e-12) or np.any(np.imag(mat) != 0):
                raise ValueError(f"Invalid: the entries of `{name}` must be probabilities in [0, 1].")
        self.prob_mat, self.pred_mat, self.select_mat = prob_mat, pred_mat, select_mat
        # The weights of the numerator and denominator of the value as linear functionals of the behaviour.
        self._win_mat = np.real(select_mat * pred_mat * prob_mat)
        self._keep_mat = np.real(select_mat * prob_mat)
        if not np.any(self._keep_mat > 0):
            raise ValueError("Invalid: no answer event with positive probability is ever kept by the referee.")
        self._classical_value = None

    @classmethod
    def hardy(cls, num_inputs: int = 2, num_outputs: int = 2) -> "PostSelectionGame":
        r"""Create the generalized Hardy game with `num_inputs` questions and `num_outputs` answers per player.

        In the generalized Hardy paradox [@meng2018hardy], every local behaviour on \(s\) questions and \(k\) answers
        per player that satisfies \(P(a < b | 0, s-1) = 0\), \(P(a < b | i, i-1) = 0\) and \(P(a > b | i-1, i-1) = 0\)
        for \(i = 1, \ldots, s-1\) also satisfies \(P(a < b | s-1, s-1) = 0\), while some quantum behaviours do not.
        The game keeps exactly the rounds whose answer event appears in one of these conditions, is won on the event
        \(a < b\) at questions \((s-1, s-1)\), and asks each of the \(2s\) question pairs involved with equal
        probability. For \(s = k = 2\) this is Hardy's original paradox [@hardy1992quantum]; the ladder paradox of
        [@boschi1997ladder] is \(k = 2\) with more questions.

        Every generalized Hardy game has classical value \(1/2\) and quantum value \(1\)
        [@callejarodriguez2026postselection]; the quantum value is approached only as the probability of post-
        selection shrinks, and for \(s = k = 2\) the largest post-selection probability of a behaviour of value one is
        \((5\sqrt{5} - 11)/8\) [@rabelo2012device].

        Args:
            num_inputs: The number of questions \(s \geq 2\) per player (default is 2).
            num_outputs: The number of answers \(k \geq 2\) per player (default is 2).

        Raises:
            ValueError: If either count is smaller than two.

        Returns:
            The generalized Hardy game.

        """
        if num_inputs < 2 or num_outputs < 2:
            raise ValueError("The generalized Hardy game needs at least two questions and two answers per player.")
        last = num_inputs - 1
        pred_mat = np.zeros((num_outputs, num_outputs, num_inputs, num_inputs))
        select_mat = np.zeros_like(pred_mat)
        for b in range(1, num_outputs):
            for a in range(b):
                pred_mat[a, b, last, last] = 1
                select_mat[a, b, last, last] = 1
                select_mat[a, b, 0, last] = 1
                for i in range(1, num_inputs):
                    select_mat[a, b, i, i - 1] = 1
                    select_mat[b, a, i - 1, i - 1] = 1
        asked = select_mat.sum(axis=(0, 1)) > 0
        return cls(asked / asked.sum(), pred_mat, select_mat)

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        """Return the value of a behaviour from its winning and post-selection weights."""
        if denominator <= 0:
            raise ValueError("The behaviour never passes post-selection, so its value is undefined.")
        return float(numerator / denominator)

    def classical_value(self, return_strategy: bool = False) -> float | tuple[float, tuple[np.ndarray, np.ndarray]]:
        r"""Compute the classical value of the post-selection game.

        The value is quasiconvex on the local polytope, so it is maximized at a deterministic strategy with positive
        post-selection probability [@callejarodriguez2026postselection]. The player with fewer deterministic
        strategies is enumerated; for each of their strategies the best response of the other player is found by
        Dinkelbach's method for the ratio, whose subproblems separate over questions, instead of by enumerating the
        other player's strategies as well.

        Args:
            return_strategy: If True, also return an optimal deterministic strategy as a pair of integer arrays
                giving Alice's answer to each question and Bob's answer to each question (default is False).

        Returns:
            The classical value, followed by the optimal strategy when `return_strategy` is True.

        """
        win, keep = self._win_mat, self._keep_mat
        alice_out, bob_out, alice_in, bob_in = win.shape
        swapped = bob_out**bob_in < alice_out**alice_in
        if swapped:
            win, keep = win.transpose(1, 0, 3, 2), keep.transpose(1, 0, 3, 2)
            alice_out, bob_out, alice_in, bob_in = win.shape
        # Axes (a, x, b, y): a strategy of the enumerated player is gathered along the first two.
        win, keep = win.transpose(0, 2, 1, 3), keep.transpose(0, 2, 1, 3)
        num_strategies = alice_out**alice_in
        chunk = max(1, _STRATEGY_CHUNK_BYTES // (2 * win.itemsize * alice_in * bob_out * bob_in))
        questions = np.arange(alice_in)
        value, alice_outputs, bob_outputs = float("-inf"), None, None
        for start in range(0, num_strategies, chunk):
            strategies = _deterministic_strategies(alice_out, alice_in, start, min(start + chunk, num_strategies))
            # wins[s, b, y] = sum_x win[strategies[s, x], x, b, y] is the game left to the other player.
            wins = win[strategies, questions].sum(axis=1)
            keeps = keep[strategies, questions].sum(axis=1)
            ratios, responses = _best_response_ratios(wins, keeps)
            best = int(ratios.argmax())
            if ratios[best] > value:
                value, alice_outputs, bob_outputs = float(ratios[best]), strategies[best], responses[best]
            if value >= 1.0:
                break
        self._classical_value = value
        if not return_strategy:
            return value
        if swapped:
            alice_outputs, bob_outputs = bob_outputs, alice_outputs
        return value, (alice_outputs, bob_outputs)

    def classical_strategy_value(self, alice_outputs: np.ndarray, bob_outputs: np.ndarray) -> float:
        """Compute the value of a deterministic strategy.

        Args:
            alice_outputs: Alice's answer to each of her questions, as an integer array of length ``num_alice_in``.
            bob_outputs: Bob's answer to each of his questions, as an integer array of length ``num_bob_in``.

        Raises:
            ValueError: If the strategy never passes post-selection.

        Returns:
            The probability of winning a kept round with these answers.

        """
        alice_in, bob_in = self.prob_mat.shape
        index = (
            np.asarray(alice_outputs)[:, np.newaxis],
            np.asarray(bob_outputs)[np.newaxis, :],
            np.arange(alice_in)[:, np.newaxis],
            np.arange(bob_in),
        )
        return self._ratio(self._win_mat[index].sum(), self._keep_mat[index].sum())

    def distribution_value(self, distribution: np.ndarray) -> float:
        """Compute the value of a conditional distribution of answers.

        Args:
            distribution: An array of shape ``(num_alice_out, num_bob_out, num_alice_in, num_bob_in)`` whose
                ``[a, b, x, y]`` entry is the probability of answers ``(a, b)`` to questions ``(x, y)``.

        Raises:
            ValueError: If the distribution never passes post-selection.

        Returns:
            The probability of winning a kept round when answering according to the distribution.

        """
        distribution = np.real(np.asarray(distribution))
        return self._ratio(np.sum(self._win_mat * distribution), np.sum(self._keep_mat * distribution))

    def post_selection_probability(self, distribution: np.ndarray) -> float:
        r"""Compute the probability \(\gamma_G(P)\) that a round is kept when answering according to a distribution.

        Args:
            distribution: An array of shape ``(num_alice_out, num_bob_out, num_alice_in, num_bob_in)`` whose
                ``[a, b, x, y]`` entry is the probability of answers ``(a, b)`` to questions ``(x, y)``.

        Returns:
            The probability that the referee keeps a round.

        """
        return float(np.sum(self._keep_mat * np.real(np.asarray(distribution))))

    def quantum_strategy_value(self, strategy: QuantumStrategy) -> float:
        r"""Compute the value of a quantum strategy.

        Args:
            strategy: A shared state on Alice's and Bob's systems and their measurements.

        Raises:
            ValueError: If the strategy never passes post-selection.

        Returns:
            The probability of winning a kept round with the behaviour
            \(P(a, b | x, y) = \mathrm{Tr}[(A_a^x \otimes B_b^y) \rho]\).

        """
        dim_alice, dim_bob = strategy.alice_povms.shape[0], strategy.bob_povms.shape[0]
        state = np.asarray(strategy.state).reshape(dim_alice, dim_bob, dim_alice, dim_bob)
        probs = np.einsum("iIxa,jJyb,IJij->abxy", strategy.alice_povms, strategy.bob_povms, state)
        return self.distribution_value(probs)

    def statistical_power(self, distribution: np.ndarray) -> float:
        r"""Compute the statistical power of a behaviour against the best local model, in bits per round.

        Playing \(n\) rounds with a behaviour \(P\) of value \(\omega_G(P)\) and post-selection probability
        \(\gamma_G(P)\), the Bayes factor and the Chernoff bound on the \(p\)-value of the hypothesis that the
        players are local both decay as \(2^{-n W_G(P)}\), where [@callejarodriguez2026postselection]

        \[
            W_G(P) = \gamma_G(P) \, D\big(\omega_G(P) \,\big\|\, \omega_\ell(G)\big)
        \]

        and \(D(p \| q) = p \log_2 (p / q) + (1 - p) \log_2 ((1 - p) / (1 - q))\) is the binary relative entropy
        to the classical value \(\omega_\ell(G)\).

        Args:
            distribution: An array of shape ``(num_alice_out, num_bob_out, num_alice_in, num_bob_in)`` whose
                ``[a, b, x, y]`` entry is the probability of answers ``(a, b)`` to questions ``(x, y)``.

        Raises:
            ValueError: If the distribution never passes post-selection.

        Returns:
            The statistical power \(W_G(P)\).

        """
        value = self.distribution_value(distribution)
        kept = self.post_selection_probability(distribution)
        local = self._classical_value if self._classical_value is not None else self.classical_value()
        divergence = xlogy(value, value) - xlogy(value, local)
        divergence += xlogy(1 - value, 1 - value) - xlogy(1 - value, 1 - local)
        return float(kept * divergence / np.log(2))

    def nonsignaling_value(
        self,
        solver: str | None = cvxpy.CLARABEL,
        solver_params: dict | None = None,
        return_strategy: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        r"""Compute the non-signaling value of the post-selection game.

        The value is invariant under scaling the behaviour, so it is maximized over the cone of unnormalized
        no-signaling behaviours with the post-selection probability fixed to one, which is a linear program.

        Args:
            solver: The cvxpy solver to use (default is Clarabel, which stays accurate when the unnormalized
                behaviour is large, as it is when the post-selection probability is small; `None` leaves the
                choice to cvxpy).
            solver_params: Extra keyword arguments passed to `cvxpy.Problem.solve`.
            return_strategy: If True, also return the optimal distribution as an array of shape
                ``(num_alice_out, num_bob_out, num_alice_in, num_bob_in)`` (default is False).

        Returns:
            The non-signaling value, followed by the optimal distribution when `return_strategy` is True.

        """
        alice_out, bob_out, alice_in, bob_in = self.pred_mat.shape
        p_var, scale, constraints = nonsignaling_cone(alice_out, bob_out, alice_in, bob_in)
        win_weights = self._win_mat.reshape(alice_out * bob_out, alice_in * bob_in)
        keep_weights = self._keep_mat.reshape(alice_out * bob_out, alice_in * bob_in)
        constraints.append(cvxpy.sum(cvxpy.multiply(keep_weights, p_var)) == 1)
        problem = cvxpy.Problem(cvxpy.Maximize(cvxpy.sum(cvxpy.multiply(win_weights, p_var))), constraints)
        value = solve_sdp(problem, solver, solver_params)
        if not return_strategy:
            return value
        return value, p_var.value.reshape(alice_out, bob_out, alice_in, bob_in) / scale.value

    def _functionals(self) -> tuple[Assemblage, cvxpy.Expression, cvxpy.Expression]:
        """Return a behaviour variable with the numerator and denominator of the value as expressions in it."""
        alice_out, bob_out, alice_in, bob_in = self.pred_mat.shape
        assemblage = Assemblage(1, alice_out, bob_out, alice_in, bob_in)
        ones = np.ones_like(self.prob_mat)
        numerator = assemblage.linear_functional(self._win_mat[np.newaxis, np.newaxis], ones)
        denominator = assemblage.linear_functional(self._keep_mat[np.newaxis, np.newaxis], ones)
        return assemblage, numerator, denominator

    @staticmethod
    def _solver_params(solver: str | None, solver_params: dict | None) -> dict:
        if solver_params is None:
            return {"eps": 1e-8, "max_iters": 100_000} if solver == cvxpy.SCS else {}
        return solver_params

    def commuting_measurement_value_upper_bound(
        self,
        k: int | str = 1,
        solver: str | None = cvxpy.CLARABEL,
        solver_params: dict | None = None,
        return_strategy: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        r"""Compute an upper bound on the commuting measurement value of the post-selection game.

        The value is invariant under scaling the behaviour, so it is maximized over the cone of unnormalized
        behaviours of the level-`k` NPA relaxation [@navascues2008convergent] with the post-selection probability
        fixed to one [@callejarodriguez2026postselection]. The level may be an integer or a string such as
        ``"1+ab"``, as in
        [`NonlocalGame.commuting_measurement_value_upper_bound`][toqito.nonlocal_games.nonlocal_game.NonlocalGame.commuting_measurement_value_upper_bound].

        The supremum need not be attained: it may be approached only as the post-selection probability goes to zero,
        in which case the numerical value is limited by the solver's range.

        Args:
            k: The level of the NPA hierarchy to use (default=1).
            solver: The cvxpy solver to use (default is Clarabel, which stays accurate when the unnormalized
                behaviour is large, as it is when the post-selection probability is small; `None` leaves the
                choice to cvxpy).
            solver_params: Extra keyword arguments passed to `cvxpy.Problem.solve`.
            return_strategy: If True, also return the optimal distribution of the relaxation, normalized, as an array
                of shape ``(num_alice_out, num_bob_out, num_alice_in, num_bob_in)`` (default is False).

        Returns:
            The upper bound, followed by the optimal distribution of the relaxation when `return_strategy` is True.

        """
        assemblage, numerator, denominator = self._functionals()
        constraints = npa_constraints(assemblage.exprs, k, normalize=False) + [denominator == 1]
        problem = cvxpy.Problem(cvxpy.Maximize(numerator), constraints)
        value = solve_sdp(problem, solver, self._solver_params(solver, solver_params))
        if not return_strategy:
            return value
        behaviour = assemblage.value()[0, 0].real
        return value, behaviour / behaviour[:, :, 0, 0].sum()

    def post_selection_probability_upper_bound(
        self,
        value: float,
        k: int | str = 1,
        solver: str | None = cvxpy.CLARABEL,
        solver_params: dict | None = None,
        return_strategy: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        r"""Compute an upper bound on the post-selection probability of commuting strategies of at least a given value.

        Among the behaviours of the level-`k` NPA relaxation [@navascues2008convergent] whose value is at least
        `value`, the post-selection probability \(\gamma_G(P)\) is maximized; the constraint on the value is linear
        once multiplied through by the post-selection probability. For a fixed value the statistical power grows
        with the post-selection probability, so scanning `value` locates the behaviour of largest statistical power
        [@callejarodriguez2026postselection]. A behaviour that is never kept satisfies the constraint trivially, so
        the result is zero when `value` exceeds the quantum value.

        Args:
            value: The smallest value the behaviour must have.
            k: The level of the NPA hierarchy to use (default=1).
            solver: The cvxpy solver to use (default is Clarabel, which stays accurate when the unnormalized
                behaviour is large, as it is when the post-selection probability is small; `None` leaves the
                choice to cvxpy).
            solver_params: Extra keyword arguments passed to `cvxpy.Problem.solve`.
            return_strategy: If True, also return the optimal distribution of the relaxation as an array of shape
                ``(num_alice_out, num_bob_out, num_alice_in, num_bob_in)`` (default is False).

        Returns:
            The upper bound on the post-selection probability, followed by the optimal distribution of the relaxation
            when `return_strategy` is True.

        """
        assemblage, numerator, denominator = self._functionals()
        constraints = npa_constraints(assemblage.exprs, k) + [numerator >= value * denominator]
        problem = cvxpy.Problem(cvxpy.Maximize(denominator), constraints)
        kept = solve_sdp(problem, solver, self._solver_params(solver, solver_params))
        if not return_strategy:
            return kept
        return kept, assemblage.value()[0, 0].real


def _best_response_ratios(wins: np.ndarray, keeps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Maximize the value over one player's deterministic strategies for a batch of the other player's strategies.

    ``wins[s, b, y]`` and ``keeps[s, b, y]`` are the winning and post-selection weights that answering ``b`` to
    question ``y`` contributes when the other player plays their strategy ``s``. For each ``s`` the ratio
    \(\sum_y w[b(y), y] / \sum_y k[b(y), y]\) is maximized over all answer functions \(b\) by Dinkelbach's method: a
    ratio \(\alpha\) can be beaten iff \(\max_b \sum_y (w - \alpha k)[b(y), y] > 0\), and that maximum separates over
    questions, so each iteration costs one argmax per question rather than a pass over all answer functions. The
    ratio is \(-\infty\) when no answer function has positive post-selection weight.

    Returns the ratios and the maximizing answer functions, with shapes ``(n,)`` and ``(n, num_in)``.
    """
    num_strategies = wins.shape[0]
    ratios = np.full(num_strategies, -np.inf)
    # Start from the answers of largest post-selection weight, which have a positive denominator whenever any do.
    responses = keeps.argmax(axis=1)
    active = keeps.reshape(num_strategies, -1).max(axis=1) > 0
    ratios[active] = 0.0
    while active.any():
        rows = np.flatnonzero(active)
        reduced = wins[rows] - ratios[rows, np.newaxis, np.newaxis] * keeps[rows]
        best = reduced.argmax(axis=1)
        gain = np.take_along_axis(reduced, best[:, np.newaxis, :], axis=1)[:, 0].sum(axis=1)
        improved = gain > _RATIO_TOL
        improved_rows, improved_best = rows[improved], best[improved][:, np.newaxis, :]
        numerators = np.take_along_axis(wins[improved_rows], improved_best, axis=1)[:, 0].sum(axis=1)
        denominators = np.take_along_axis(keeps[improved_rows], improved_best, axis=1)[:, 0].sum(axis=1)
        ratios[improved_rows] = numerators / denominators
        responses[improved_rows] = best[improved]
        active[rows[~improved]] = False
    return ratios, responses
