"""Two-player nonlocal game."""

import itertools
from dataclasses import dataclass

import cvxpy
import numpy as np

from toqito._sdp import solve_sdp
from toqito.matrix_ops import tensor
from toqito.nonlocal_games.binary_constraint_system_game import (
    _parity_system_from_truth_tables,
    check_perfect_commuting_strategy,
)
from toqito.rand import random_povm, random_unitary
from toqito.state_opt.npa_hierarchy import npa_constraints


def _random_measurements(dim: int, num_inputs: int, num_outputs: int, seed: int) -> np.ndarray:
    """Draw random starting measurements for a see-saw, as an array indexed by ``[input, output]``.

    When the dimension allows, each measurement is projective in a Haar-random basis (the surplus basis vectors
    all go to the last outcome); such starts reach entangled optima more reliably than generic POVMs. Otherwise a
    random POVM is drawn.
    """
    if dim < num_outputs:
        return random_povm(dim, num_inputs, num_outputs, seed=seed).transpose(2, 3, 0, 1)
    rng = np.random.default_rng(seed)
    measurements = np.zeros((num_inputs, num_outputs, dim, dim), dtype=complex)
    for y in range(num_inputs):
        basis = random_unitary(dim, seed=int(rng.integers(0, 2**32 - 1)))
        for column in range(dim):
            vector = basis[:, column]
            measurements[y, min(column, num_outputs - 1)] += np.outer(vector, vector.conj())
    return measurements


def _validate_game(prob_mat: np.ndarray, pred_mat: np.ndarray, pred_ndim: int) -> None:
    """Check that the predicate has the expected shape and that the question distribution matches it."""
    if pred_mat.ndim != pred_ndim:
        raise ValueError(f"Invalid: `pred_mat` must have {pred_ndim} dimensions, got {pred_mat.ndim}.")
    if prob_mat.shape != pred_mat.shape[-2:]:
        raise ValueError(
            f"Invalid: `prob_mat` has shape {prob_mat.shape} but `pred_mat` has {pred_mat.shape[-2:]} question pairs."
        )
    if np.min(prob_mat) < -1e-12:
        raise ValueError(
            "Invalid: The variable `prob_mat` must be a probability matrix: its entries must be non-negative."
        )
    if not np.isclose(np.sum(prob_mat), 1, atol=1e-8):
        raise ValueError("Invalid: The variable `prob_mat` must be a probability matrix: its entries must sum to 1.")


@dataclass(frozen=True)
class QuantumStrategy:
    """A shared state together with one measurement per question for each player.

    Attributes:
        state: The shared density operator, on Alice's system tensored with Bob's for a nonlocal game, and on the
            referee's system tensored with Alice's and then Bob's for an extended nonlocal game.
        alice_povms: Alice's measurements as an array of shape ``(dim_alice, dim_alice, num_alice_in, num_alice_out)``
            whose ``[:, :, x, a]`` slice is the operator for answer ``a`` to question ``x``.
        bob_povms: Bob's measurements, laid out in the same way with shape ``(dim_bob, dim_bob, num_bob_in,
            num_bob_out)``.

    """

    state: np.ndarray
    alice_povms: np.ndarray
    bob_povms: np.ndarray


def _strategy_from_seesaw(rho_vals: np.ndarray, bob_vals: np.ndarray, referee_dim: int = 1) -> QuantumStrategy:
    r"""Recover a quantum strategy from the operators the see-saw optimizes over.

    The see-saw represents Alice's side by the operators \(\rho_a^x = \mathrm{Tr}_A[(A_a^x \otimes \mathbb{1})\rho]\)
    on the referee and Bob, with \(\tau = \sum_a \rho_a^x\) their reduced state. Taking the purification
    \(|\psi\rangle = \sum_{ij} \overline{\sqrt{\tau}}_{ij} |i\rangle|j\rangle\) on a copy of that space for Alice gives
    \(\mathrm{Tr}_A[(M \otimes \mathbb{1})|\psi\rangle\langle\psi|] = \sqrt{\tau} M^{\mathsf{T}} \sqrt{\tau}\), so
    \(A_a^x = (\tau^{-1/2} \rho_a^x \tau^{-1/2})^{\mathsf{T}}\) reproduces every \(\rho_a^x\). The inverse is taken on
    the support of \(\tau\) and the complement of the support is added to one element of each POVM.

    Args:
        rho_vals: Array of shape ``(num_alice_in, num_alice_out, d, d)`` holding \(\rho_a^x\).
        bob_vals: Array of shape ``(num_bob_in, num_bob_out, dim_bob, dim_bob)`` holding Bob's POVMs.
        referee_dim: The referee dimension, with ``d = referee_dim * dim_bob``.

    """
    alice_in, alice_out, joint_dim, _ = rho_vals.shape
    dim_bob = joint_dim // referee_dim
    tau = rho_vals[0].sum(axis=0)
    tau = (tau + tau.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(tau)
    support = eigvals > 1e-10 * eigvals.max()
    sqrt_tau = (eigvecs[:, support] * np.sqrt(eigvals[support])) @ eigvecs[:, support].conj().T
    inv_sqrt_tau = (eigvecs[:, support] / np.sqrt(eigvals[support])) @ eigvecs[:, support].conj().T
    complement = np.eye(joint_dim) - eigvecs[:, support] @ eigvecs[:, support].conj().T

    alice_povms = np.einsum("ij,xajk,kl->lixa", inv_sqrt_tau, rho_vals, inv_sqrt_tau)  # transposed in the last step
    alice_povms[:, :, :, 0] += complement.T[:, :, np.newaxis]

    # |psi> on Alice's copy tensored with (referee, Bob); reorder to (referee, Alice, Bob) for an extended game.
    psi = sqrt_tau.conj().reshape(joint_dim, referee_dim, dim_bob).transpose(1, 0, 2).reshape(-1)
    state = np.outer(psi, psi.conj())
    return QuantumStrategy(state=state, alice_povms=alice_povms, bob_povms=bob_vals.transpose(2, 3, 0, 1))


class NonlocalGame:
    r"""Create two-player nonlocal game object.

    *Nonlocal games* are a mathematical framework that abstractly models a
    physical system. This game is played between two players, Alice and Bob, who
    are not allowed to communicate with each other once the game has started and
    who play cooperative against an adversary referred to as the referee.

    The nonlocal game framework was originally introduced in [@cleve2010consequences].

    A tutorial is available in the documentation. For more info, see
    [Nonlocal Games](../../../generated/gallery/nonlocal_games/nonlocal_game.md).

    """

    def __init__(self, prob_mat: np.ndarray, pred_mat: np.ndarray, reps: int = 1) -> None:
        """Construct nonlocal game object.

        Args:
            prob_mat: A matrix whose (x, y)-entry gives the probability
                      that the referee will give Alice the value `x` and Bob
                      the value `y`.
            pred_mat: A four-dimensional matrix whose (a,b,x,y)-entry gives
                      the outcome for answers "a" and "b" given questions
                      "x" and "y".
            reps: Number of parallel repetitions to perform. Default is 1.

        Raises:
            ValueError: If `pred_mat` is not four-dimensional, if its question axes do not match the shape of
                `prob_mat`, or if `prob_mat` is not a probability distribution.

        """
        prob_mat = np.asarray(prob_mat)
        pred_mat = np.asarray(pred_mat)
        _validate_game(prob_mat, pred_mat, pred_ndim=4)
        self.reps = reps
        if reps == 1:
            self.prob_mat = prob_mat
            self.pred_mat = pred_mat
        else:
            num_alice_out, num_bob_out, num_alice_in, num_bob_in = pred_mat.shape
            self.prob_mat = tensor(prob_mat, reps)

            pred_mat2 = np.zeros(
                (
                    num_alice_out**reps,
                    num_bob_out**reps,
                    num_alice_in**reps,
                    num_bob_in**reps,
                )
            )
            for i, i_ind in enumerate(itertools.product(range(num_alice_in), repeat=reps)):
                for j, j_ind in enumerate(itertools.product(range(num_bob_in), repeat=reps)):
                    to_tensor = np.empty([reps, num_alice_out, num_bob_out])
                    for k in range(reps - 1, -1, -1):
                        to_tensor[k] = pred_mat[:, :, i_ind[k], j_ind[k]]
                    pred_mat2[:, :, i, j] = tensor(to_tensor)
            self.pred_mat = pred_mat2
        # The truth-table constraints this game was built from, when it came from `from_bcs_game`.
        self._raw_constraints = None

    @classmethod
    def from_bcs_game(cls, constraints: list[np.ndarray], reps: int = 1) -> "NonlocalGame":
        r"""Convert constraints that specify a binary constraint system game to a nonlocal game.

        Binary constraint system games (BCS) games were originally defined in [@cleve2014characterization].

        Each constraint is a 0/1 truth table of shape ``(2,) * n`` over the ``n`` binary variables, with a 1 at
        every assignment that satisfies it. The referee sends Alice a constraint and Bob one of its variables; they
        win when Alice's assignment satisfies the constraint and agrees with Bob's bit. `create_bcs_constraints`
        builds such truth tables for parity constraints.

        Args:
            constraints: List of truth tables, one per constraint, all of shape ``(2,) * n``.
            reps: Number of parallel repetitions to perform. Default is 1.

        Raises:
            ValueError: If no constraints are given, if the constraints do not all have shape ``(2,) * n``, or if
                some constraint does not depend on any variable.

        Returns:
            A NonlocalGame object arising from the variables and constraints that define the game.

        """
        if (num_constraints := len(constraints)) == 0:
            raise ValueError("At least 1 constraint is required")
        constraints = [np.asarray(constraint) for constraint in constraints]
        num_variables = constraints[0].ndim
        if any(constraint.shape != (2,) * num_variables for constraint in constraints):
            raise ValueError(f"Each constraint must be a truth table of shape {(2,) * num_variables}.")

        # Constraint j depends on variable i when its truth table is not constant along axis i. Bob is only ever
        # asked about a variable the constraint depends on, uniformly at random.
        dependent_variables = np.array(
            [[np.diff(constraint, axis=i).any() for i in range(num_variables)] for constraint in constraints],
            dtype=float,
        )
        num_dependent = dependent_variables.sum(axis=1)
        if (degenerate := np.flatnonzero(num_dependent == 0)).size:
            raise ValueError(f"Constraint {degenerate[0]} is degenerate (has no dependent variables).")
        prob_mat = dependent_variables / num_dependent[:, np.newaxis] / num_constraints

        # Alice's answer `a` is an assignment, read as the binary digits of `a` with the most significant bit first,
        # which is the flat index of the truth table. Bob's answer must be the bit of that assignment at his variable.
        assignments = np.indices((2,) * num_variables).reshape(num_variables, -1).T
        satisfied = np.stack([constraint.reshape(-1) == 1 for constraint in constraints], axis=1)
        pred_mat = np.zeros((2**num_variables, 2, num_constraints, num_variables))
        for y_ques in range(num_variables):
            pred_mat[np.arange(2**num_variables), assignments[:, y_ques], :, y_ques] = satisfied
        game = cls(prob_mat, pred_mat, reps)
        game._raw_constraints = constraints
        return game

    def is_bcs_perfect_commuting_strategy(self) -> bool:
        r"""Determine if the BCS game admits a perfect commuting-operator strategy.

        The game must have been built by `from_bcs_game` from parity constraints; the criterion of
        `check_perfect_commuting_strategy` is then applied to the recovered linear system.

        Raises:
            ValueError: If the game was not created from a BCS game, or if some constraint is not a parity constraint.

        Returns:
            True if a perfect commuting-operator strategy exists; False otherwise.

        """
        if self._raw_constraints is None:
            raise ValueError("This game was not constructed from a binary constraint system; use `from_bcs_game`.")
        M, b = _parity_system_from_truth_tables(self._raw_constraints)
        return check_perfect_commuting_strategy(M, b)

    def classical_value(self, return_strategy: bool = False) -> float | tuple[float, tuple[np.ndarray, np.ndarray]]:
        r"""Compute the classical value of the nonlocal game.

        Args:
            return_strategy: If True, also return an optimal deterministic strategy as a pair of integer arrays
                giving Alice's answer to each question and Bob's answer to each question (default is False).

        Returns:
            The classical value, followed by the optimal strategy when `return_strategy` is True.

        """
        A_out, B_out, A_in, B_in = self.pred_mat.shape
        pm = self.pred_mat * self.prob_mat

        # Enumerate the smaller of the two deterministic-strategy spaces.
        swapped = A_out**A_in < B_out**B_in
        if swapped:
            pm = pm.transpose((1, 0, 3, 2))
            A_out, B_out, A_in, B_in = pm.shape

        # Reorder axes to (A_out, A_in, B_out, B_in).
        pm = pm.transpose((0, 2, 1, 3))

        # Each deterministic strategy for Bob assigns an output b(y) to every question y. For a fixed Bob strategy,
        # Alice independently picks the best output a for each question x, so the winning probability is
        # sum_x max_a sum_y pm[a, x, b(y), y]. The classical value is the maximum over all of Bob's strategies.
        strategies = np.array(list(itertools.product(range(B_out), repeat=B_in)))  # (num_strategies, B_in)
        # gathered[a, x, s, y] = pm[a, x, strategies[s, y], y]
        gathered = pm[:, :, strategies, np.arange(B_in)]  # (A_out, A_in, num_strategies, B_in)
        per_question = gathered.sum(axis=-1)  # (A_out, A_in, num_strategies)
        per_strategy = per_question.max(axis=0).sum(axis=0)  # (num_strategies,)
        best = int(per_strategy.argmax())
        value = float(per_strategy[best])
        if not return_strategy:
            return value
        bob_outputs = strategies[best]
        alice_outputs = per_question[:, :, best].argmax(axis=0)
        if swapped:
            alice_outputs, bob_outputs = bob_outputs, alice_outputs
        return value, (alice_outputs, bob_outputs)

    def classical_strategy_value(self, alice_outputs: np.ndarray, bob_outputs: np.ndarray) -> float:
        """Compute the winning probability of a deterministic strategy.

        Args:
            alice_outputs: Alice's answer to each of her questions, as an integer array of length ``num_alice_in``.
            bob_outputs: Bob's answer to each of his questions, as an integer array of length ``num_bob_in``.

        Returns:
            The probability of winning the game with these answers.

        """
        alice_in, bob_in = self.prob_mat.shape
        alice_outputs = np.asarray(alice_outputs)[:, np.newaxis]
        bob_outputs = np.asarray(bob_outputs)[np.newaxis, :]
        wins = self.pred_mat[alice_outputs, bob_outputs, np.arange(alice_in)[:, np.newaxis], np.arange(bob_in)]
        return float(np.real(np.sum(self.prob_mat * wins)))

    def quantum_strategy_value(self, strategy: QuantumStrategy) -> float:
        r"""Compute the winning probability of a quantum strategy.

        Args:
            strategy: A shared state on Alice's and Bob's systems and their measurements.

        Returns:
            The probability \(\sum_{x,y,a,b} \pi(x,y) V(a,b|x,y) \mathrm{Tr}[(A_a^x \otimes B_b^y) \rho]\).

        """
        dim_alice, dim_bob = strategy.alice_povms.shape[0], strategy.bob_povms.shape[0]
        state = np.asarray(strategy.state).reshape(dim_alice, dim_bob, dim_alice, dim_bob)
        probs = np.einsum("iIxa,jJyb,IJij->abxy", strategy.alice_povms, strategy.bob_povms, state)
        return float(np.real(np.sum(self.pred_mat * self.prob_mat * probs)))

    def distribution_value(self, distribution: np.ndarray) -> float:
        """Compute the winning probability of a conditional distribution of answers.

        Args:
            distribution: An array of shape ``(num_alice_out, num_bob_out, num_alice_in, num_bob_in)`` whose
                ``[a, b, x, y]`` entry is the probability of answers ``(a, b)`` to questions ``(x, y)``.

        Returns:
            The probability of winning the game when answering according to the distribution.

        """
        return float(np.real(np.sum(self.pred_mat * self.prob_mat * np.asarray(distribution))))

    def quantum_value_lower_bound(
        self,
        dim: int = 2,
        iters: int = 5,
        tol: float = 1e-6,
        seed: int | None = None,
        max_steps: int = 100,
        solver: str | None = cvxpy.SCS,
        solver_params: dict | None = None,
        return_strategy: bool = False,
    ) -> float | tuple[float, QuantumStrategy]:
        r"""Compute a lower bound on the quantum value of a nonlocal game [@liang2007bounds].

        Calculates a lower bound on the maximum value that the specified
        nonlocal game can take on in quantum mechanical settings where Alice and
        Bob each have access to `dim`-dimensional quantum system.

        This function works by starting with a randomly-generated POVM for Bob,
        and then optimizing Alice's POVM and the shared entangled state. Then
        Alice's POVM and the entangled state are fixed, and Bob's POVM is
        optimized. And so on, back and forth between Alice and Bob until
        convergence is reached.

        Note that the algorithm is not guaranteed to obtain the optimal local
        bound and can get stuck in local minimum values. The alleviate this, the
        `iters` parameter allows one to run the algorithm some pre-specified
        number of times and keep the highest value obtained.

        The algorithm is based on the alternating projections algorithm as it
        can be applied to Bell inequalities as shown in [@liang2007bounds].

        The alternating projection algorithm has also been referred to as the
        "see-saw" algorithm as it goes back and forth between the following two
        semidefinite programs:

        \[
            \begin{equation}
                \begin{aligned}
                    \textbf{SDP-1:} \quad & \\
                    \text{maximize:} \quad & \sum_{(x,y \in \Sigma)} \pi(x,y)
                                             \sum_{(a,b) \in \Gamma}
                                             V(a,b|x,y)
                                             \langle B_b^y, A_a^x \rangle \\
                    \text{subject to:} \quad & \sum_{a \in \Gamma_{\mathsf{A}}}=
                                        \tau, \qquad \qquad
                                        \forall x \in \Sigma_{\mathsf{A}}, \\
                                       \quad & A_a^x \in \text{Pos}(\mathcal{A}),
                                        \qquad
                                        \forall x \in \Sigma_{\mathsf{A}}, \
                                        \forall a \in \Gamma_{\mathsf{A}}, \\
                                        & \tau \in \text{D}(\mathcal{A}).
                \end{aligned}
            \end{equation}
        \]

        \[
            \begin{equation}
                \begin{aligned}
                    \textbf{SDP-2:} \quad & \\
                    \text{maximize:} \quad & \sum_{(x,y \in \Sigma)} \pi(x,y)
                                             \sum_{(a,b) \in \Gamma} V(a,b|x,y)
                                             \langle B_b^y, A_a^x \rangle \\
                    \text{subject to:} \quad & \sum_{b \in \Gamma_{\mathsf{B}}}=
                                        \mathbb{I}, \qquad \qquad
                                        \forall y \in \Sigma_{\mathsf{B}}, \\
                                    \quad & B_b^y \in \text{Pos}(\mathcal{B}),
                                    \qquad \forall y \in \Sigma_{\mathsf{B}}, \
                                    \forall b \in \Gamma_{\mathsf{B}}.
                \end{aligned}
            \end{equation}
        \]

        Args:
            dim: The dimension of the quantum system that Alice and Bob have
                    access to (default = 2).
            iters: The number of random restarts of the alternating projection algorithm (default = 5).
            tol: The see-saw stops once the winning probability changes by at most `tol` between
                    consecutive steps (default = 1e-6).
            seed: Optional seed for reproducibly generating the random starting
                    POVMs (default is None).
            max_steps: The maximum number of see-saw steps per restart (default = 100).
            solver: The cvxpy solver to use (default is SCS).
            solver_params: Extra keyword arguments passed to `cvxpy.Problem.solve` (default, for SCS, is
                    ``{"eps": 1e-8, "max_iters": 50_000}``, tight enough that the returned value is an achievable
                    winning probability to within solver accuracy).
            return_strategy: If True, also return the strategy attaining the bound as a `QuantumStrategy` on
                    two `dim`-dimensional systems (default is False).

        Returns:
            The lower bound on the quantum value of a nonlocal game, followed by the strategy attaining it when
            `return_strategy` is True.

        Examples:
            The CHSH game

            The CHSH game is a two-player nonlocal game with the following
            probability distribution and question and answer sets.

            \[
                \begin{equation}
                \begin{aligned}
                    \pi(x,y) = \frac{1}{4}, \qquad (x,y) \in \Sigma_A \times \Sigma_B,
                    \qquad \text{and} \qquad (a, b) \in \Gamma_A \times \Gamma_B,
                \end{aligned}
                \end{equation}
            \]

            where

            \[
                \begin{equation}
                \Sigma_A = \{0, 1\}, \quad \Sigma_B = \{0, 1\}, \quad \Gamma_A =
                \{0,1\}, \quad \text{and} \quad \Gamma_B = \{0, 1\}.
                \end{equation}
            \]

            Alice and Bob win the CHSH game if and only if the following equation is
            satisfied.

            \[
                \begin{equation}
                a \oplus b = x \land y.
                \end{equation}
            \]

            Recall that \(\oplus\) refers to the XOR operation.

            The optimal quantum value of CHSH is
            \(\cos(\pi/8)^2 \approx 0.8536\) where the optimal classical value
            is \(3/4\).

            ```python exec="1" source="above" result="text"
            import numpy as np
            from toqito.nonlocal_games.nonlocal_game import NonlocalGame

            dim = 2
            num_alice_inputs, num_alice_outputs = 2, 2
            num_bob_inputs, num_bob_outputs = 2, 2
            prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
            pred_mat = np.zeros((num_alice_outputs, num_bob_outputs, num_alice_inputs, num_bob_inputs))

            for a_alice in range(num_alice_outputs):
                for b_bob in range(num_bob_outputs):
                    for x_alice in range(num_alice_inputs):
                        for y_bob in range(num_bob_inputs):
                            if np.mod(a_alice + b_bob + x_alice * y_bob, dim) == 0:
                                pred_mat[a_alice, b_bob, x_alice, y_bob] = 1

            chsh = NonlocalGame(prob_mat, pred_mat)

            print(chsh.quantum_value_lower_bound())
            ```

        """
        num_outputs_alice, num_outputs_bob, num_inputs_alice, num_inputs_bob = self.pred_mat.shape
        if solver_params is None:
            solver_params = {"eps": 1e-8, "max_iters": 50_000} if solver == cvxpy.SCS else {}
        # weights[a, b, x, y] = pi(x, y) V(a, b | x, y)
        weights = self.pred_mat * self.prob_mat
        rng = np.random.default_rng(seed)

        # Both alternating SDPs have the same variables and constraints on every see-saw step and restart; only the
        # fixed opponent operators change. Each is built once with the opponent folded into one `cvxpy.Parameter`
        # per own operator, so a see-saw step only reassigns parameter values and re-solves.

        # SDP-1: Alice's operators A_a^x, with sum_a A_a^x = tau for every x. The objective is
        # sum_{x, a} <C_a^x, A_a^x> with C_a^x = sum_{y, b} pi(x, y) V(a, b | x, y) B_b^y.
        alice_ops = {
            (x, a): cvxpy.Variable((dim, dim), hermitian=True)
            for x in range(num_inputs_alice)
            for a in range(num_outputs_alice)
        }
        tau = cvxpy.Variable((dim, dim), hermitian=True)
        alice_coeffs = {key: cvxpy.Parameter((dim, dim), complex=True) for key in alice_ops}
        alice_win = sum(cvxpy.real(cvxpy.trace(alice_coeffs[key] @ alice_ops[key])) for key in alice_ops)
        alice_constraints = [op >> 0 for op in alice_ops.values()]
        alice_constraints += [
            sum(alice_ops[x, a] for a in range(num_outputs_alice)) == tau for x in range(num_inputs_alice)
        ]
        # tau is positive semidefinite as a sum of positive semidefinite operators, so only its trace is constrained.
        alice_constraints.append(cvxpy.trace(tau) == 1)
        alice_problem = cvxpy.Problem(cvxpy.Maximize(alice_win), alice_constraints)

        # SDP-2: Bob's POVMs B_b^y, with sum_b B_b^y = I for every y. The objective is
        # sum_{y, b} <D_b^y, B_b^y> with D_b^y = sum_{x, a} pi(x, y) V(a, b | x, y) A_a^x.
        bob_ops = {
            (y, b): cvxpy.Variable((dim, dim), hermitian=True)
            for y in range(num_inputs_bob)
            for b in range(num_outputs_bob)
        }
        bob_coeffs = {key: cvxpy.Parameter((dim, dim), complex=True) for key in bob_ops}
        bob_win = sum(cvxpy.real(cvxpy.trace(bob_coeffs[key] @ bob_ops[key])) for key in bob_ops)
        bob_constraints = [op >> 0 for op in bob_ops.values()]
        bob_constraints += [
            sum(bob_ops[y, b] for b in range(num_outputs_bob)) == np.identity(dim) for y in range(num_inputs_bob)
        ]
        bob_problem = cvxpy.Problem(cvxpy.Maximize(bob_win), bob_constraints)

        best_lower_bound = float("-inf")
        best_operators = None
        for _ in range(iters):
            # Random measurements for Bob are the starting point of the alternating projection algorithm.
            bob_vals = _random_measurements(dim, num_inputs_bob, num_outputs_bob, int(rng.integers(0, 2**32 - 1)))

            prev_win = float("-inf")
            for _step in range(max_steps):
                coeffs = np.einsum("abxy,ybij->xaij", weights, bob_vals)
                for (x, a), param in alice_coeffs.items():
                    param.value = coeffs[x, a]
                solve_sdp(alice_problem, solver, solver_params)
                alice_vals = np.array(
                    [[alice_ops[x, a].value for a in range(num_outputs_alice)] for x in range(num_inputs_alice)]
                )

                coeffs = np.einsum("abxy,xaij->ybij", weights, alice_vals)
                for (y, b), param in bob_coeffs.items():
                    param.value = coeffs[y, b]
                lower_bound = solve_sdp(bob_problem, solver, solver_params)
                bob_vals = np.array(
                    [[bob_ops[y, b].value for b in range(num_outputs_bob)] for y in range(num_inputs_bob)]
                )

                if lower_bound > best_lower_bound:
                    best_lower_bound, best_operators = lower_bound, (alice_vals, bob_vals)
                # Use the absolute change so a non-monotone step does not terminate the loop prematurely.
                if abs(lower_bound - prev_win) <= tol:
                    break
                prev_win = lower_bound

        if not return_strategy:
            return best_lower_bound
        return best_lower_bound, _strategy_from_seesaw(*best_operators)

    def nonsignaling_value(
        self, solver: str | None = None, solver_params: dict | None = None, return_strategy: bool = False
    ) -> float | tuple[float, np.ndarray]:
        r"""Compute the non-signaling value of the nonlocal game.

        The non-signaling value is the maximum winning probability over all conditional distributions
        \(p(a, b | x, y)\) whose marginals do not depend on the other player's question, which is a linear program.

        Args:
            solver: The cvxpy solver to use (default is cvxpy's choice).
            solver_params: Extra keyword arguments passed to `cvxpy.Problem.solve`.
            return_strategy: If True, also return the optimal distribution as an array of shape
                ``(num_alice_out, num_bob_out, num_alice_in, num_bob_in)`` (default is False).

        Returns:
            A value between `[0, 1]` representing the non-signaling value, followed by the optimal distribution when
            `return_strategy` is True.

        """
        alice_out, bob_out, alice_in, bob_in = self.pred_mat.shape

        # p_var[a * bob_out + b, x * bob_in + y] is the probability of answers (a, b) to questions (x, y), laid out as
        # a matrix so that marginals are matrix products.
        p_var = cvxpy.Variable((alice_out * bob_out, alice_in * bob_in), nonneg=True)
        # The marginals of Alice and Bob, which non-signaling requires to be independent of the other question.
        alice_marginal = cvxpy.Variable((alice_out, alice_in), nonneg=True)
        bob_marginal = cvxpy.Variable((bob_out, bob_in), nonneg=True)

        sum_over_b = np.kron(np.eye(alice_out), np.ones((1, bob_out)))
        sum_over_a = np.kron(np.ones((1, alice_out)), np.eye(bob_out))
        repeat_over_y = np.kron(np.eye(alice_in), np.ones((1, bob_in)))
        repeat_over_x = np.kron(np.ones((1, alice_in)), np.eye(bob_in))
        constraints = [
            sum_over_b @ p_var == alice_marginal @ repeat_over_y,
            sum_over_a @ p_var == bob_marginal @ repeat_over_x,
            cvxpy.sum(alice_marginal, axis=0) == 1,
            cvxpy.sum(bob_marginal, axis=0) == 1,
        ]

        weights = np.real(self.pred_mat * self.prob_mat).reshape(alice_out * bob_out, alice_in * bob_in)
        problem = cvxpy.Problem(cvxpy.Maximize(cvxpy.sum(cvxpy.multiply(weights, p_var))), constraints)
        value = solve_sdp(problem, solver, solver_params)
        if not return_strategy:
            return value
        return value, p_var.value.reshape(alice_out, bob_out, alice_in, bob_in)

    def commuting_measurement_value_upper_bound(
        self,
        k: int | str = 1,
        solver: str | None = None,
        solver_params: dict | None = None,
        return_strategy: bool = False,
    ) -> float | tuple[float, np.ndarray]:
        r"""Compute an upper bound on the commuting measurement value of the nonlocal game.

        This function calculates an upper bound on the commuting measurement value by
            using k-levels of the NPA hierarchy [@navascues2008convergent]. The NPA hierarchy is a uniform
            family of semidefinite programs that converges to the commuting measurement value of
            any nonlocal game.

        You can determine the level of the hierarchy by a positive integer or a string
            of a form like '1+ab+aab', which indicates that an intermediate level of the hierarchy
            should be used, where this example uses all products of one measurement, all products of
            one Alice and one Bob measurement, and all products of two Alice and one Bob measurements.

        Args:
            k: The level of the NPA hierarchy to use (default=1).
            solver: The cvxpy solver to use (default is cvxpy's choice).
            solver_params: Extra keyword arguments passed to `cvxpy.Problem.solve`.
            return_strategy: If True, also return the optimal distribution of the relaxation as an array of shape
                ``(num_alice_out, num_bob_out, num_alice_in, num_bob_in)`` (default is False). At a finite level
                this distribution need not be realisable by commuting measurements.

        Returns:
            The upper bound on the commuting strategy value of a nonlocal game, followed by the optimal distribution
            of the relaxation when `return_strategy` is True.

        """
        alice_out, bob_out, alice_in, bob_in = self.pred_mat.shape
        weights = np.real(self.pred_mat * self.prob_mat)

        mat = {
            (x, y): cvxpy.Variable((alice_out, bob_out), name=f"M(a, b | {x}, {y})")
            for x in range(alice_in)
            for y in range(bob_in)
        }
        p_win = sum(cvxpy.sum(cvxpy.multiply(weights[:, :, x, y], mat[x, y])) for (x, y) in mat)
        problem = cvxpy.Problem(cvxpy.Maximize(p_win), npa_constraints(mat, k))
        value = solve_sdp(problem, solver, solver_params)
        if not return_strategy:
            return value
        distribution = np.zeros((alice_out, bob_out, alice_in, bob_in))
        for (x, y), var in mat.items():
            distribution[:, :, x, y] = var.value
        return value, distribution
