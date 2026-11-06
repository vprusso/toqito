"""Two-player extended nonlocal game."""

import itertools
from typing import Literal, TypeAlias

import cvxpy
import numpy as np

from toqito._sdp import solve_sdp
from toqito.matrix_ops import tensor
from toqito.nonlocal_games.nonlocal_game import _random_measurements, _validate_game
from toqito.state_opt.npa_hierarchy import npa_constraints

AnswerEventConstraint: TypeAlias = tuple[
    dict[tuple[int, int, int, int], float] | np.ndarray,
    Literal["==", "<=", ">="],
    float,
]


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
        _validate_game(prob_mat, pred_mat, pred_ndim=6)
        if pred_mat.shape[0] != pred_mat.shape[1]:
            raise ValueError("Invalid: the referee operators in `pred_mat` must be square.")
        self.reps = reps
        if reps == 1:
            self.prob_mat = prob_mat
            self.pred_mat = pred_mat
        else:
            referee_dim, _, num_alice_out, num_bob_out, num_alice_in, num_bob_in = pred_mat.shape
            self.prob_mat = tensor(prob_mat, reps)

            pred_mat2 = np.zeros(
                (
                    referee_dim**reps,
                    referee_dim**reps,
                    num_alice_out**reps,
                    num_bob_out**reps,
                    num_alice_in**reps,
                    num_bob_in**reps,
                ),
                dtype=pred_mat.dtype,
            )
            for i, i_ind in enumerate(itertools.product(range(num_alice_in), repeat=reps)):
                for j, j_ind in enumerate(itertools.product(range(num_bob_in), repeat=reps)):
                    to_tensor = np.empty(
                        [reps, referee_dim, referee_dim, num_alice_out, num_bob_out], dtype=pred_mat.dtype
                    )
                    for k in range(reps - 1, -1, -1):
                        to_tensor[k] = pred_mat[:, :, :, :, i_ind[k], j_ind[k]]
                    pred_mat2[:, :, :, :, i, j] = tensor(to_tensor)
            self.pred_mat = pred_mat2
        (
            self.referee_dim,
            _,
            self.num_alice_out,
            self.num_bob_out,
            self.num_alice_in,
            self.num_bob_in,
        ) = self.pred_mat.shape

    def unentangled_value(self) -> float:
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

        Returns:
            The unentangled value of the extended nonlocal game.

        """
        alice_out, bob_out, alice_in, bob_in = self.num_alice_out, self.num_bob_out, self.num_alice_in, self.num_bob_in
        # weights[a, b, x, y] is the referee operator pi(x, y) V(a, b | x, y).
        weights = np.moveaxis(self.pred_mat * self.prob_mat, (0, 1), (-2, -1))

        alice_strategies = np.array(list(itertools.product(range(alice_out), repeat=alice_in)))  # (num_f, alice_in)
        max_unent_val = float("-inf")
        for g_bob in itertools.product(range(bob_out), repeat=bob_in):
            # summed_y[a, x] = sum_y weights[a, g(y), x, y]
            summed_y = weights[:, list(g_bob), :, np.arange(bob_in)].sum(axis=0)
            # operators[f] = sum_x summed_y[f(x), x], for every deterministic strategy f of Alice at once
            operators = summed_y[alice_strategies, np.arange(alice_in)].sum(axis=1)
            max_unent_val = max(max_unent_val, np.linalg.eigvalsh(operators).max())
        return float(max_unent_val)

    def nonsignaling_value(self, solver: str | None = None, solver_params: dict | None = None) -> float:
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

        Returns:
            The non-signaling value of the extended nonlocal game.

        """
        dim = self.referee_dim
        alice_out, bob_out, alice_in, bob_in = self.num_alice_out, self.num_bob_out, self.num_alice_in, self.num_bob_in

        # cvxpy variables are at most two-dimensional, so the operator-valued functions are dictionaries of
        # `dim`-by-`dim` Hermitian variables indexed by answers and questions.
        k_var = {
            (a, b, x, y): cvxpy.Variable((dim, dim), hermitian=True)
            for a in range(alice_out)
            for b in range(bob_out)
            for x in range(alice_in)
            for y in range(bob_in)
        }
        sigma = {(a, x): cvxpy.Variable((dim, dim), hermitian=True) for a in range(alice_out) for x in range(alice_in)}
        rho = {(b, y): cvxpy.Variable((dim, dim), hermitian=True) for b in range(bob_out) for y in range(bob_in)}
        tau = cvxpy.Variable((dim, dim), hermitian=True)

        p_win = sum(
            self.prob_mat[x, y] * cvxpy.trace(self.pred_mat[:, :, a, b, x, y].conj().T @ var)
            for (a, b, x, y), var in k_var.items()
        )

        constraints = [var >> 0 for var in k_var.values()]
        # The non-signaling conditions: Alice's marginal does not depend on y and Bob's does not depend on x.
        constraints += [
            sum(k_var[a, b, x, y] for b in range(bob_out)) == sigma[a, x]
            for a in range(alice_out)
            for x in range(alice_in)
            for y in range(bob_in)
        ]
        constraints += [
            sum(k_var[a, b, x, y] for a in range(alice_out)) == rho[b, y]
            for b in range(bob_out)
            for x in range(alice_in)
            for y in range(bob_in)
        ]
        # Both marginals sum to the referee's reduced state tau, a density operator.
        constraints += [sum(sigma[a, x] for a in range(alice_out)) == tau for x in range(alice_in)]
        constraints += [sum(rho[b, y] for b in range(bob_out)) == tau for y in range(bob_in)]
        constraints += [cvxpy.trace(tau) == 1, tau >> 0]

        problem = cvxpy.Problem(cvxpy.Maximize(cvxpy.real(p_win)), constraints)
        return solve_sdp(problem, solver, solver_params)

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
    ) -> float:
        r"""Calculate lower bound on the quantum value of an extended nonlocal game.

        Uses the see-saw method of `NonlocalGame.quantum_value_lower_bound`: Bob's measurements are fixed while an SDP
        optimizes Alice's measurements and the shared state (represented by the operators
        \(\rho_a^x = \mathrm{Tr}_A[(A_a^x \otimes \mathbb{1}) \rho]\) on the referee and Bob), then those are fixed
        while an SDP optimizes Bob's measurements, and so on until the winning probability stops improving. The
        procedure is restarted from `iters` random measurements for Bob and the best value found is returned.

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

        Raises:
            ValueError: If `initial_bob_povms` does not have shape ``(dim, dim, num_bob_in, num_bob_out)``.

        Returns:
            The best lower bound found on the quantum value.

        """
        referee_dim = self.referee_dim
        alice_out, bob_out, alice_in, bob_in = self.num_alice_out, self.num_bob_out, self.num_alice_in, self.num_bob_in
        if dim is None:
            dim = bob_out
        if solver_params is None:
            solver_params = {"eps": 1e-8, "max_iters": 50_000} if solver == cvxpy.SCS else {}
        if initial_bob_povms is not None and np.shape(initial_bob_povms) != (dim, dim, bob_in, bob_out):
            raise ValueError(f"`initial_bob_povms` must have shape {(dim, dim, bob_in, bob_out)}.")
        rng = np.random.default_rng(seed)
        joint_dim = referee_dim * dim

        # Both SDPs are built once; the fixed opponent operators enter through `cvxpy.Parameter`s whose values are
        # recomputed in numpy at every see-saw step.

        # Alice's SDP: maximize sum_{x, a} <C_a^x, rho_a^x> over rho_a^x >= 0 with sum_a rho_a^x = tau and
        # Tr(tau) = 1, where C_a^x = sum_{y, b} pi(x, y) V(a, b | x, y) (x) B_b^y.
        rho_vars = {
            (x, a): cvxpy.Variable((joint_dim, joint_dim), hermitian=True)
            for x in range(alice_in)
            for a in range(alice_out)
        }
        tau = cvxpy.Variable((joint_dim, joint_dim), hermitian=True)
        rho_coeffs = {key: cvxpy.Parameter((joint_dim, joint_dim), complex=True) for key in rho_vars}
        alice_win = sum(cvxpy.real(cvxpy.trace(rho_coeffs[key] @ rho_vars[key])) for key in rho_vars)
        alice_constraints = [var >> 0 for var in rho_vars.values()]
        alice_constraints += [sum(rho_vars[x, a] for a in range(alice_out)) == tau for x in range(alice_in)]
        # tau is positive semidefinite as a sum of positive semidefinite operators, so only its trace is constrained.
        alice_constraints.append(cvxpy.trace(tau) == 1)
        alice_problem = cvxpy.Problem(cvxpy.Maximize(alice_win), alice_constraints)

        # Bob's SDP: maximize sum_{y, b} <D_b^y, B_b^y> over POVMs {B_b^y}_b, where
        # D_b^y = sum_{x, a} pi(x, y) Tr_R[(V(a, b | x, y) (x) 1) rho_a^x].
        bob_vars = {(y, b): cvxpy.Variable((dim, dim), hermitian=True) for y in range(bob_in) for b in range(bob_out)}
        bob_coeffs = {key: cvxpy.Parameter((dim, dim), complex=True) for key in bob_vars}
        bob_win = sum(cvxpy.real(cvxpy.trace(bob_coeffs[key] @ bob_vars[key])) for key in bob_vars)
        bob_constraints = [var >> 0 for var in bob_vars.values()]
        bob_constraints += [sum(bob_vars[y, b] for b in range(bob_out)) == np.identity(dim) for y in range(bob_in)]
        bob_problem = cvxpy.Problem(cvxpy.Maximize(bob_win), bob_constraints)

        best_lower_bound = float("-inf")
        for restart in range(iters):
            if restart == 0 and initial_bob_povms is not None:
                bob_vals = np.asarray(initial_bob_povms, dtype=complex).transpose(2, 3, 0, 1)
            else:
                bob_vals = _random_measurements(dim, bob_in, bob_out, int(rng.integers(0, 2**32 - 1)))

            prev_win = float("-inf")
            for _step in range(max_steps):
                coeffs = np.einsum("xy,ijabxy,ybkl->xaikjl", self.prob_mat, self.pred_mat, bob_vals)
                coeffs = coeffs.reshape(alice_in, alice_out, joint_dim, joint_dim)
                for (x, a), param in rho_coeffs.items():
                    param.value = coeffs[x, a]
                solve_sdp(alice_problem, solver, solver_params)
                rho_vals = np.array([[rho_vars[x, a].value for a in range(alice_out)] for x in range(alice_in)])
                rho_vals = rho_vals.reshape(alice_in, alice_out, referee_dim, dim, referee_dim, dim)

                coeffs = np.einsum("xy,ikabxy,xaklij->yblj", self.prob_mat, self.pred_mat, rho_vals)
                for (y, b), param in bob_coeffs.items():
                    param.value = coeffs[y, b]
                lower_bound = solve_sdp(bob_problem, solver, solver_params)
                bob_vals = np.array([[bob_vars[y, b].value for b in range(bob_out)] for y in range(bob_in)])

                best_lower_bound = max(best_lower_bound, lower_bound)
                if abs(lower_bound - prev_win) <= tol:
                    break
                prev_win = lower_bound

        return best_lower_bound

    @staticmethod
    def _answer_event_probability(
        assemblage: dict[tuple[int, int], cvxpy.Variable],
        a: int,
        b: int,
        x: int,
        y: int,
        referee_dim: int,
    ) -> cvxpy.Expression:
        r"""Return the commuting-measurement answer probability \(p(a, b \mid x, y)\).

        At NPA level 1 and above, the block
        \(K_{xy}(a,b) = \mathrm{Tr}_{AB}(I_R \otimes A_a^x B_b^y \, \sigma)\)
        in the assemblage satisfies \(p(a,b \mid x,y) = \mathrm{Tr}\, K_{xy}(a,b)\).
        """
        block = assemblage[x, y][
            a * referee_dim : (a + 1) * referee_dim,
            b * referee_dim : (b + 1) * referee_dim,
        ]
        return cvxpy.trace(block)

    @staticmethod
    def _answer_event_linear_constraint(
        assemblage: dict[tuple[int, int], cvxpy.Variable],
        constraint: AnswerEventConstraint,
        referee_dim: int,
        num_a_out: int,
        num_b_out: int,
        num_a_in: int,
        num_b_in: int,
    ) -> cvxpy.constraints.constraint.Constraint:
        """Build a single cvxpy constraint from sparse or dense answer-event coefficients."""
        coeffs, op, rhs = constraint
        terms: list[cvxpy.Expression] = []
        expected_shape = (num_a_out, num_b_out, num_a_in, num_b_in)

        if isinstance(coeffs, np.ndarray):
            if coeffs.shape != expected_shape:
                raise ValueError(f"Dense constraint coefficients must have shape {expected_shape}, got {coeffs.shape}.")
            for x in range(num_a_in):
                for y in range(num_b_in):
                    for a in range(num_a_out):
                        for b in range(num_b_out):
                            coeff = float(coeffs[a, b, x, y])
                            if coeff != 0.0:
                                terms.append(
                                    coeff
                                    * ExtendedNonlocalGame._answer_event_probability(
                                        assemblage, a, b, x, y, referee_dim
                                    )
                                )
        else:
            # Sparse keys are validated explicitly with per-index bounds.
            for (a, b, x, y), coeff in coeffs.items():
                if not (0 <= a < num_a_out and 0 <= b < num_b_out and 0 <= x < num_a_in and 0 <= y < num_b_in):
                    raise ValueError(
                        f"Constraint index (a={a}, b={b}, x={x}, y={y}) is out of range for game dimensions "
                        f"{expected_shape}."
                    )
                terms.append(
                    float(coeff) * ExtendedNonlocalGame._answer_event_probability(assemblage, a, b, x, y, referee_dim)
                )

        linear_expr = cvxpy.real(cvxpy.sum(terms)) if terms else cvxpy.Constant(0)

        if op == "==":
            return linear_expr == rhs
        if op == "<=":
            return linear_expr <= rhs
        if op == ">=":
            return linear_expr >= rhs
        raise ValueError(f"Constraint operator must be '==', '<=', or '>=', got {op!r}.")

    def commuting_measurement_value_upper_bound(
        self,
        k: int | str = 1,
        no_signaling: bool = True,
        constraints: list[AnswerEventConstraint] | None = None,
        solver: str | None = cvxpy.SCS,
        solver_params: dict | None = None,
    ) -> float:
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

        Returns:
            The upper bound on the commuting strategy value of an extended nonlocal game.

        """
        if solver_params is None:
            solver_params = {"eps": 1e-8, "max_iters": 100_000} if solver == cvxpy.SCS else {}
        dR, _, A_out, B_out, A_in, B_in = self.pred_mat.shape
        K = {}
        for x in range(A_in):
            for y in range(B_in):
                blocks = [[None for _ in range(B_out)] for _ in range(A_out)]
                for a in range(A_out):
                    for b in range(B_out):
                        blocks[a][b] = cvxpy.Variable((dR, dR), hermitian=True, name=f"K({x},{y})_{a}{b}")
                K[(x, y)] = cvxpy.bmat(blocks)
        # The winning probability is sum_{x, y} <W_xy, K_xy> with W_xy the block matrix of weighted referee operators
        # pi(x, y) V(a, b | x, y), laid out in the same (a, b) block pattern as the assemblage K_xy.
        total_win = 0
        for x in range(A_in):
            for y in range(B_in):
                blocks = [[self.pred_mat[:, :, a, b, x, y] for b in range(B_out)] for a in range(A_out)]
                weights = self.prob_mat[x, y] * np.block(blocks)
                total_win += cvxpy.sum(cvxpy.multiply(weights.conj(), K[(x, y)]))
        cons = npa_constraints(K, k, referee_dim=dR, no_signaling=no_signaling)
        if constraints is not None:
            for constraint in constraints:
                cons.append(self._answer_event_linear_constraint(K, constraint, dR, A_out, B_out, A_in, B_in))
        prob = cvxpy.Problem(cvxpy.Maximize(cvxpy.real(total_win)), cons)
        return solve_sdp(prob, solver, solver_params)
