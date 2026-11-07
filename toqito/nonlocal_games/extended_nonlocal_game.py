"""Two-player extended nonlocal game."""

import itertools

import cvxpy
import numpy as np

from toqito._sdp import solve_sdp
from toqito.nonlocal_games._common import (
    AnswerEventConstraint,
    Assemblage,
    answer_event_linear_constraint,
    answer_event_probability,
    npa_upper_bound,
    parallel_repetition,
    seesaw,
    validate_game,
)
from toqito.nonlocal_games.nonlocal_game import QuantumStrategy, _strategy_from_seesaw

__all__ = ["AnswerEventConstraint", "ExtendedNonlocalGame"]


class ExtendedNonlocalGame:
    r"""Create two-player extended nonlocal game object.

    *Extended nonlocal games* are a superset of nonlocal games in which the
    players share a tripartite state with the referee. In such games, the
    winning conditions for Alice and Bob may depend on outcomes of measurements
    made by the referee, on its part of the shared quantum state, in addition
    to Alice and Bob's answers to the questions sent by the referee.

    Extended nonlocal games were initially defined in [@johnston2016extended] and more
    information on these games can be found in [@russo2017extended].

    For a detailed walkthrough and several examples, including the BB84 and CHSH
    games, please see the tutorial on
    [Extended Nonlocal Games](../../../generated/gallery/extended_nonlocal_games/enlg_introduction.md).

    """

    def __init__(self, prob_mat: np.ndarray, pred_mat: np.ndarray, reps: int = 1) -> None:
        """Construct extended nonlocal game object.

        Args:
            prob_mat: A matrix whose (x, y)-entry gives the probability that the referee will give Alice the value `x`
                and Bob the value `y`.
            pred_mat: A six-dimensional array whose ``[:, :, a, b, x, y]`` slice is the referee's operator for answers
                ``(a, b)`` to questions ``(x, y)``.
            reps: Number of parallel repetitions to perform.

        Raises:
            ValueError: If `pred_mat` is not six-dimensional with square referee operators, if its question axes do
                not match the shape of `prob_mat`, or if `prob_mat` is not a probability distribution.

        """
        prob_mat = np.asarray(prob_mat)
        pred_mat = np.asarray(pred_mat)
        validate_game(prob_mat, pred_mat, pred_ndim=6)
        if pred_mat.shape[0] != pred_mat.shape[1]:
            raise ValueError("Invalid: the referee operators in `pred_mat` must be square.")
        self.reps = reps
        self.prob_mat, self.pred_mat = (
            parallel_repetition(prob_mat, pred_mat, reps) if reps > 1 else (prob_mat, pred_mat)
        )
        (
            self.referee_dim,
            _,
            self.num_alice_out,
            self.num_bob_out,
            self.num_alice_in,
            self.num_bob_in,
        ) = self.pred_mat.shape

    def unentangled_strategy_value(
        self, alice_outputs: np.ndarray, bob_outputs: np.ndarray, referee_state: np.ndarray
    ) -> float:
        """Compute the winning probability of an unentangled strategy.

        Args:
            alice_outputs: Alice's answer to each of her questions, as an integer array of length ``num_alice_in``.
            bob_outputs: Bob's answer to each of his questions, as an integer array of length ``num_bob_in``.
            referee_state: The density operator the players prepare on the referee's system.

        Returns:
            The probability of winning the game with these answers and this referee state.

        """
        alice_in, bob_in = self.prob_mat.shape
        alice_outputs = np.asarray(alice_outputs)[:, np.newaxis]
        bob_outputs = np.asarray(bob_outputs)[np.newaxis, :]
        operators = self.pred_mat[
            :, :, alice_outputs, bob_outputs, np.arange(alice_in)[:, np.newaxis], np.arange(bob_in)
        ]
        total = np.einsum("xy,ijxy->ij", self.prob_mat, operators)
        return float(np.real(np.trace(total @ np.asarray(referee_state))))

    def quantum_strategy_value(self, strategy: QuantumStrategy) -> float:
        r"""Compute the winning probability of a quantum strategy.

        Args:
            strategy: A shared state on the referee's, Alice's and Bob's systems (in that order) and the players'
                measurements.

        Returns:
            The probability \(\sum_{x,y,a,b} \pi(x,y) \mathrm{Tr}[(V(a,b|x,y) \otimes A_a^x \otimes B_b^y) \rho]\).

        """
        dim_alice, dim_bob = strategy.alice_povms.shape[0], strategy.bob_povms.shape[0]
        dims = (self.referee_dim, dim_alice, dim_bob)
        state = np.asarray(strategy.state).reshape(dims + dims)
        total = np.einsum("rRabxy,iIxa,jJyb,RIJrij->xy", self.pred_mat, strategy.alice_povms, strategy.bob_povms, state)
        return float(np.real(np.sum(self.prob_mat * total)))

    def assemblage_value(self, assemblage: np.ndarray) -> float:
        r"""Compute the winning probability of an assemblage.

        Args:
            assemblage: An array of shape ``(referee_dim, referee_dim, num_alice_out, num_bob_out, num_alice_in,
                num_bob_in)`` whose ``[:, :, a, b, x, y]`` slice is the referee's unnormalised state conditioned on
                answers ``(a, b)`` to questions ``(x, y)``.

        Returns:
            The probability \(\sum_{x,y,a,b} \pi(x,y) \langle V(a,b|x,y), K(a,b|x,y) \rangle\).

        """
        overlaps = np.einsum("ijabxy,ijabxy->xy", self.pred_mat.conj(), np.asarray(assemblage))
        return float(np.real(np.sum(self.prob_mat * overlaps)))

    def unentangled_value(
        self, return_strategy: bool = False
    ) -> float | tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        r"""Calculate the unentangled value of an extended nonlocal game.

        The *unentangled value* of an extended nonlocal game is the supremum
        value for Alice and Bob's winning probability in the game over all
        unentangled strategies. Due to convexity and compactness, it is possible
        to calculate the unentangled extended nonlocal game by:

        \[
            \omega(G) = \max_{f, g}
            \lVert
            \sum_{(x,y) \in \Sigma_A \times \Sigma_B} \pi(x,y)
            V(f(x), g(y)|x, y)
            \rVert
        \]

        where the maximum is over all functions \(f : \Sigma_A \rightarrow
        \Gamma_A\) and \(g : \Sigma_B \rightarrow \Gamma_B\).

        Args:
            return_strategy: If True, also return an optimal unentangled strategy as a triple of Alice's answers to
                her questions, Bob's answers to his, and the pure referee state they prepare (as a density operator)
                (default is False).

        Returns:
            The unentangled value of the extended nonlocal game, followed by the optimal strategy when
            `return_strategy` is True.

        """
        alice_out, bob_out, alice_in, bob_in = self.num_alice_out, self.num_bob_out, self.num_alice_in, self.num_bob_in
        # weights[a, b, x, y] is the referee operator pi(x, y) V(a, b | x, y).
        weights = np.moveaxis(self.pred_mat * self.prob_mat, (0, 1), (-2, -1))

        alice_strategies = np.array(list(itertools.product(range(alice_out), repeat=alice_in)))  # (num_f, alice_in)
        max_unent_val = float("-inf")
        best = None
        for g_bob in itertools.product(range(bob_out), repeat=bob_in):
            # summed_y[a, x] = sum_y weights[a, g(y), x, y]
            summed_y = weights[:, list(g_bob), :, np.arange(bob_in)].sum(axis=0)
            # operators[f] = sum_x summed_y[f(x), x], for every deterministic strategy f of Alice at once
            operators = summed_y[alice_strategies, np.arange(alice_in)].sum(axis=1)
            largest = np.linalg.eigvalsh(operators)[:, -1]
            f_best = int(largest.argmax())
            if largest[f_best] > max_unent_val:
                max_unent_val = float(largest[f_best])
                best = (alice_strategies[f_best], np.array(g_bob), operators[f_best])
        if not return_strategy:
            return max_unent_val
        alice_outputs, bob_outputs, operator = best
        # The referee state is the top eigenvector of the winning operator of the best strategy.
        referee_vector = np.linalg.eigh(operator)[1][:, -1]
        return max_unent_val, (alice_outputs, bob_outputs, np.outer(referee_vector, referee_vector.conj()))

    def nonsignaling_value(
        self, solver: str | None = None, solver_params: dict | None = None, return_strategy: bool = False
    ) -> float | tuple[float, np.ndarray]:
        r"""Calculate the non-signaling value of an extended nonlocal game.

        The *non-signaling value* of an extended nonlocal game is the supremum
        value of the winning probability of the game taken over all
        non-signaling strategies for Alice and Bob.

        A *non-signaling strategy* for an extended nonlocal game consists of a
        function

        \[
            K : \Gamma_A \times \Gamma_B \times \Sigma_A \times \Sigma_B
            \rightarrow \text{Pos}(\mathcal{R})
        \]

        such that

        \[
            \sum_{a \in \Gamma_A} K(a,b|x,y) = \rho_b^y
            \quad \text{and} \quad
            \sum_{b \in \Gamma_B} K(a,b|x,y) = \sigma_a^x,
        \]

        for all \(x \in \Sigma_A\) and \(y \in \Sigma_B\) where
        \(\{\rho_b^y : y \in \Sigma_A, \ b \in \Gamma_B\}\) and
        \(\{\sigma_a^x : x \in \Sigma_A, \ a \in \Gamma_B\}\) are
        collections of operators satisfying

        \[
            \sum_{a \in \Gamma_A} \rho_b^y =
            \tau =
            \sum_{b \in \Gamma_B} \sigma_a^x,
        \]

        for every choice of \(x \in \Sigma_A\) and \(y \in \Sigma_B\)
        where \(\tau \in \text{D}(\mathcal{R})\) is a density operator.

        Args:
            solver: The cvxpy solver to use (default is cvxpy's choice).
            solver_params: Extra keyword arguments passed to `cvxpy.Problem.solve`.
            return_strategy: If True, also return the optimal non-signaling assemblage \(K\) as an array of shape
                ``(referee_dim, referee_dim, num_alice_out, num_bob_out, num_alice_in, num_bob_in)`` (default is
                False).

        Returns:
            The non-signaling value of the extended nonlocal game, followed by the optimal assemblage when
            `return_strategy` is True.

        """
        dim = self.referee_dim
        alice_out, bob_out, alice_in, bob_in = self.num_alice_out, self.num_bob_out, self.num_alice_in, self.num_bob_in

        assemblage = Assemblage(dim, alice_out, bob_out, alice_in, bob_in)
        k_mat = assemblage.exprs
        p_win = assemblage.linear_functional(self.pred_mat, self.prob_mat)

        # `sum_a @ K_xy` is the row of blocks sum_a K(a, b | x, y) and `K_xy @ sum_b.T` the column of blocks
        # sum_b K(a, b | x, y).
        sum_a = np.kron(np.ones((1, alice_out)), np.eye(dim))
        sum_b = np.kron(np.ones((1, bob_out)), np.eye(dim))

        constraints = [
            k_mat[x, y][a * dim : (a + 1) * dim, b * dim : (b + 1) * dim] >> 0
            for x in range(alice_in)
            for y in range(bob_in)
            for a in range(alice_out)
            for b in range(bob_out)
        ]
        # The non-signaling conditions: Bob's marginal does not depend on x and Alice's does not depend on y. They
        # make the sum of all blocks, the referee's reduced state tau, the same for every question pair.
        for y in range(bob_in):
            reference = sum_a @ k_mat[0, y]
            constraints += [sum_a @ k_mat[x, y] == reference for x in range(1, alice_in)]
        for x in range(alice_in):
            reference = k_mat[x, 0] @ sum_b.T
            constraints += [k_mat[x, y] @ sum_b.T == reference for y in range(1, bob_in)]
        constraints.append(cvxpy.real(cvxpy.trace(sum_a @ k_mat[0, 0] @ sum_b.T)) == 1)

        problem = cvxpy.Problem(cvxpy.Maximize(p_win), constraints)
        value = solve_sdp(problem, solver, solver_params)
        if not return_strategy:
            return value
        return value, assemblage.value()

    def quantum_value_lower_bound(
        self,
        dim: int | None = None,
        iters: int = 5,
        tol: float = 1e-6,
        seed: int | None = None,
        max_steps: int = 100,
        initial_bob_povms: np.ndarray | None = None,
        solver: str | None = cvxpy.SCS,
        solver_params: dict | None = None,
        return_strategy: bool = False,
    ) -> float | tuple[float, QuantumStrategy]:
        r"""Calculate lower bound on the quantum value of an extended nonlocal game.

        Uses the see-saw method of `NonlocalGame.quantum_value_lower_bound`: Bob's measurements are fixed while an SDP
        optimizes Alice's measurements and the shared state (represented by the operators
        \(\rho_a^x = \mathrm{Tr}_A[(A_a^x \otimes \mathbb{1}) \rho]\) on the referee and Bob), then those are fixed
        while an SDP optimizes Bob's measurements (in closed form when Bob has two answers), and so on until the
        winning probability stops improving. The procedure is restarted from `iters` random measurements for Bob
        and the best value found is returned.

        Args:
            dim: The dimension of Bob's quantum system (default is the number of Bob's outputs).
            iters: The number of random restarts of the see-saw (default is 5).
            tol: The see-saw stops once the winning probability changes by at most `tol` between consecutive steps
                (default is 1e-6).
            seed: Optional seed for reproducibly generating the random starting POVMs (default is None).
            max_steps: The maximum number of see-saw steps per restart (default is 100).
            initial_bob_povms: Optional measurements for Bob to start the first restart from, as an array of shape
                ``(dim, dim, num_bob_in, num_bob_out)`` in the layout of `toqito.rand.random_povm`.
            solver: The cvxpy solver to use (default is SCS).
            solver_params: Extra keyword arguments passed to `cvxpy.Problem.solve` (default, for SCS, is
                ``{"eps": 1e-8, "max_iters": 50_000}``, tight enough that the returned value is an achievable
                winning probability to within solver accuracy).
            return_strategy: If True, also return the strategy attaining the bound as a `QuantumStrategy` whose state
                lives on the referee's system, Alice's system of dimension ``referee_dim * dim`` and Bob's system of
                dimension ``dim``, in that order (default is False).

        Raises:
            ValueError: If `initial_bob_povms` does not have shape ``(dim, dim, num_bob_in, num_bob_out)``.

        Returns:
            The best lower bound found on the quantum value, followed by the strategy attaining it when
            `return_strategy` is True.

        """
        if dim is None:
            dim = self.num_bob_out
        value, rho_vals, bob_vals = seesaw(
            self.prob_mat, self.pred_mat, dim, iters, tol, seed, max_steps, initial_bob_povms, solver, solver_params
        )
        if not return_strategy:
            return value
        return value, _strategy_from_seesaw(rho_vals, bob_vals, referee_dim=self.referee_dim)

    _answer_event_probability = staticmethod(answer_event_probability)
    _answer_event_linear_constraint = staticmethod(answer_event_linear_constraint)

    def commuting_measurement_value_upper_bound(
        self,
        k: int | str = 1,
        no_signaling: bool = True,
        constraints: list[AnswerEventConstraint] | None = None,
        solver: str | None = cvxpy.SCS,
        solver_params: dict | None = None,
        return_strategy: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        r"""Compute an upper bound on the commuting measurement value of an extended nonlocal game.

        This function calculates an upper bound on the commuting measurement value by
            using k-levels of the NPA hierarchy [@navascues2008convergent]. The NPA hierarchy is a uniform
            family of semidefinite programs that converges to the commuting measurement value of
            any extended nonlocal game.

        You can determine the level of the hierarchy by a positive integer or a string
            of a form like '1+ab+aab', which indicates that an intermediate level of the hierarchy
            should be used, where this example uses all products of one measurement, all products of
            one Alice and one Bob measurement, and all products of two Alice and one Bob measurements.

        Optional linear constraints on answer-event probabilities may be supplied. Each
        constraint is a triple ``(coeffs, op, rhs)`` where ``coeffs`` is either a sparse
        dictionary mapping ``(a, b, x, y)`` to a real coefficient or a dense array of shape
        ``(num_alice_out, num_bob_out, num_alice_in, num_bob_in)``, ``op`` is one of
        ``"=="``, ``"<="``, or ``">="``, and ``rhs`` is a float. The constraint is

        \[
            \sum_{a,b,x,y} c_{abxy}\, p(a,b \mid x,y) \;\operatorname{op}\; d,
        \]

        with \(p(a,b \mid x,y) = \mathrm{Tr}\, K_{xy}(a,b)\) at the NPA level used
        (level 1 and above expose these marginals on the assemblage blocks).

        Args:
            k: The level of the NPA hierarchy to use (default=1).
            no_signaling: Whether to enforce the no-signaling constraints (default=True).
            constraints: Optional linear constraints on answer-event probabilities.
            solver: The cvxpy solver to use (default is SCS).
            solver_params: Extra keyword arguments passed to `cvxpy.Problem.solve` (default, for SCS, is
                ``{"eps": 1e-8, "max_iters": 100_000}``).
            return_strategy: If True, also return the optimal assemblage of the relaxation as an array of shape
                ``(referee_dim, referee_dim, num_alice_out, num_bob_out, num_alice_in, num_bob_in)`` (default is
                False). At a finite level this assemblage need not be realisable by commuting measurements.

        Returns:
            The upper bound on the commuting strategy value of an extended nonlocal game, followed by the optimal
            assemblage of the relaxation when `return_strategy` is True.

        """
        value, assemblage = npa_upper_bound(
            self.prob_mat, self.pred_mat, k, no_signaling, constraints, solver, solver_params
        )
        return (value, assemblage) if return_strategy else value
