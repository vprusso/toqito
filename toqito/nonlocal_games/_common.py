"""Building blocks shared by nonlocal games and extended nonlocal games.

A nonlocal game is an extended nonlocal game whose referee has a one-dimensional system, so every routine here takes
the predicate in its six-dimensional form ``pred_mat[r, s, a, b, x, y]`` and a nonlocal game passes
``pred_mat[np.newaxis, np.newaxis]``.
"""

import itertools
from typing import Literal, TypeAlias

import cvxpy
import numpy as np
import scipy.sparse

from toqito._sdp import solve_sdp
from toqito.matrix_ops import tensor
from toqito.rand import random_povm, random_unitary
from toqito.state_opt.npa_hierarchy import npa_constraints

AnswerEventConstraint: TypeAlias = tuple[
    dict[tuple[int, int, int, int], float] | np.ndarray,
    Literal["==", "<=", ">="],
    float,
]


def validate_game(prob_mat: np.ndarray, pred_mat: np.ndarray, pred_ndim: int) -> None:
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


def parallel_repetition(prob_mat: np.ndarray, pred_mat: np.ndarray, reps: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the question distribution and predicate of ``reps`` parallel repetitions of a game.

    The predicate's last two axes index the questions; all leading axes (answers, and the referee operator's indices
    for an extended game) are tensored across repetitions, so each becomes ``reps`` times as long in the output.
    """
    *leading, alice_in, bob_in = pred_mat.shape
    repeated = np.zeros([d**reps for d in leading] + [alice_in**reps, bob_in**reps], dtype=pred_mat.dtype)
    for i, i_ind in enumerate(itertools.product(range(alice_in), repeat=reps)):
        for j, j_ind in enumerate(itertools.product(range(bob_in), repeat=reps)):
            repeated[..., i, j] = tensor(np.stack([pred_mat[..., i_ind[k], j_ind[k]] for k in range(reps)]))
    return tensor(prob_mat, reps), repeated


def nonsignaling_cone(
    alice_out: int, bob_out: int, alice_in: int, bob_in: int
) -> tuple[cvxpy.Variable, cvxpy.Variable, list[cvxpy.constraints.constraint.Constraint]]:
    """Return a variable for an unnormalized no-signaling behaviour, its total weight and the constraints of the cone.

    ``p_var[a * bob_out + b, x * bob_in + y]`` is the weight of answers (a, b) to questions (x, y), laid out as a
    matrix so that marginals are matrix products. Each player's marginal is independent of the other player's
    question, and every question pair carries the same total weight ``scale``; adding ``scale == 1`` gives the
    no-signaling polytope.
    """
    p_var = cvxpy.Variable((alice_out * bob_out, alice_in * bob_in), nonneg=True)
    alice_marginal = cvxpy.Variable((alice_out, alice_in), nonneg=True)
    bob_marginal = cvxpy.Variable((bob_out, bob_in), nonneg=True)
    scale = cvxpy.Variable(nonneg=True)
    sum_over_b = np.kron(np.eye(alice_out), np.ones((1, bob_out)))
    sum_over_a = np.kron(np.ones((1, alice_out)), np.eye(bob_out))
    repeat_over_y = np.kron(np.eye(alice_in), np.ones((1, bob_in)))
    repeat_over_x = np.kron(np.ones((1, alice_in)), np.eye(bob_in))
    constraints = [
        sum_over_b @ p_var == alice_marginal @ repeat_over_y,
        sum_over_a @ p_var == bob_marginal @ repeat_over_x,
        cvxpy.sum(alice_marginal, axis=0) == scale,
        cvxpy.sum(bob_marginal, axis=0) == scale,
    ]
    return p_var, scale, constraints


def random_measurements(dim: int, num_inputs: int, num_outputs: int, seed: int) -> np.ndarray:
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


def best_two_outcome_measurements(coeffs: np.ndarray) -> tuple[float, np.ndarray]:
    r"""Maximise \(\sum_y \sum_{b} \langle D_b^y, B_b^y \rangle\) over two-outcome measurements \(\{B_0^y, B_1^y\}\).

    For each ``y`` the objective is \(\langle D_1^y, \mathbb{1} \rangle + \langle D_0^y - D_1^y, B_0^y \rangle\)
    with \(0 \leq B_0^y \leq \mathbb{1}\), which is maximised by the projector onto the positive eigenspace of the
    Hermitian part of \(D_0^y - D_1^y\).

    Args:
        coeffs: The operators \(D_b^y\) as an array of shape ``(num_inputs, 2, dim, dim)``.

    Returns:
        The maximum and the measurements attaining it, as an array of the same shape as ``coeffs``.

    """
    difference = coeffs[:, 0] - coeffs[:, 1]
    difference = (difference + difference.conj().transpose(0, 2, 1)) / 2
    eigvals, eigvecs = np.linalg.eigh(difference)
    positive = eigvecs * (eigvals > 0)[:, np.newaxis, :]
    first = positive @ positive.conj().transpose(0, 2, 1)
    measurements = np.stack([first, np.eye(coeffs.shape[-1]) - first], axis=1)
    value = float(np.einsum("ybij,ybji->", coeffs, measurements).real)
    return value, measurements


def seesaw(
    prob_mat: np.ndarray,
    pred_mat: np.ndarray,
    dim: int,
    iters: int,
    tol: float,
    seed: int | None,
    max_steps: int,
    initial_bob_povms: np.ndarray | None,
    solver: str | None,
    solver_params: dict | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    r"""Run the see-saw lower bound on the quantum value and return the best value with the operators attaining it.

    Bob's measurements are fixed while an SDP optimizes Alice's side, represented by the operators
    \(\rho_a^x = \mathrm{Tr}_A[(A_a^x \otimes \mathbb{1})\rho]\) on the referee and Bob with
    \(\sum_a \rho_a^x = \tau\) and \(\mathrm{Tr}(\tau) = 1\); then those are fixed while an SDP optimizes Bob's
    measurements, and so on until the winning probability changes by at most ``tol``. The procedure is restarted from
    ``iters`` random measurements for Bob (the first restart uses ``initial_bob_povms`` when given).

    Returns the best winning probability found, the operators \(\rho_a^x\) as an array of shape
    ``(num_alice_in, num_alice_out, referee_dim * dim, referee_dim * dim)`` and Bob's POVMs as an array of shape
    ``(num_bob_in, num_bob_out, dim, dim)``.
    """
    referee_dim, _, alice_out, bob_out, alice_in, bob_in = pred_mat.shape
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
    # D_b^y = sum_{x, a} pi(x, y) Tr_R[(V(a, b | x, y) (x) 1) rho_a^x]. With two answers the optimum is the
    # projective measurement onto the positive and negative parts of D_0^y - D_1^y, so no SDP is built.
    if bob_out != 2:
        bob_vars = {(y, b): cvxpy.Variable((dim, dim), hermitian=True) for y in range(bob_in) for b in range(bob_out)}
        bob_coeffs = {key: cvxpy.Parameter((dim, dim), complex=True) for key in bob_vars}
        bob_win = sum(cvxpy.real(cvxpy.trace(bob_coeffs[key] @ bob_vars[key])) for key in bob_vars)
        bob_constraints = [var >> 0 for var in bob_vars.values()]
        bob_constraints += [sum(bob_vars[y, b] for b in range(bob_out)) == np.identity(dim) for y in range(bob_in)]
        bob_problem = cvxpy.Problem(cvxpy.Maximize(bob_win), bob_constraints)

    best_value = float("-inf")
    best_operators = None
    for restart in range(iters):
        if restart == 0 and initial_bob_povms is not None:
            bob_vals = np.asarray(initial_bob_povms, dtype=complex).transpose(2, 3, 0, 1)
        else:
            bob_vals = random_measurements(dim, bob_in, bob_out, int(rng.integers(0, 2**32 - 1)))

        prev_win = float("-inf")
        for _step in range(max_steps):
            coeffs = np.einsum("xy,ijabxy,ybkl->xaikjl", prob_mat, pred_mat, bob_vals)
            coeffs = coeffs.reshape(alice_in, alice_out, joint_dim, joint_dim)
            for (x, a), param in rho_coeffs.items():
                param.value = coeffs[x, a]
            solve_sdp(alice_problem, solver, solver_params)
            rho_vals = np.array([[rho_vars[x, a].value for a in range(alice_out)] for x in range(alice_in)])
            rho_blocks = rho_vals.reshape(alice_in, alice_out, referee_dim, dim, referee_dim, dim)

            coeffs = np.einsum("xy,ikabxy,xaklij->yblj", prob_mat, pred_mat, rho_blocks)
            if bob_out == 2:
                value, bob_vals = best_two_outcome_measurements(coeffs)
            else:
                for (y, b), param in bob_coeffs.items():
                    param.value = coeffs[y, b]
                value = solve_sdp(bob_problem, solver, solver_params)
                bob_vals = np.array([[bob_vars[y, b].value for b in range(bob_out)] for y in range(bob_in)])

            if value > best_value:
                best_value, best_operators = value, (rho_vals, bob_vals)
            # Use the absolute change so a non-monotone step does not terminate the loop prematurely.
            if abs(value - prev_win) <= tol:
                break
            prev_win = value

    return best_value, *best_operators


def answer_event_probability(
    assemblage: dict[tuple[int, int], cvxpy.Expression], a: int, b: int, x: int, y: int, referee_dim: int
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


def answer_event_linear_constraint(
    assemblage: dict[tuple[int, int], cvxpy.Expression],
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
                            terms.append(coeff * answer_event_probability(assemblage, a, b, x, y, referee_dim))
    else:
        # Sparse keys are validated explicitly with per-index bounds.
        for (a, b, x, y), coeff in coeffs.items():
            if not (0 <= a < num_a_out and 0 <= b < num_b_out and 0 <= x < num_a_in and 0 <= y < num_b_in):
                raise ValueError(
                    f"Constraint index (a={a}, b={b}, x={x}, y={y}) is out of range for game dimensions "
                    f"{expected_shape}."
                )
            terms.append(float(coeff) * answer_event_probability(assemblage, a, b, x, y, referee_dim))

    linear_expr = cvxpy.real(cvxpy.sum(terms)) if terms else cvxpy.Constant(0)

    if op == "==":
        return linear_expr == rhs
    if op == "<=":
        return linear_expr <= rhs
    if op == ">=":
        return linear_expr >= rhs
    raise ValueError(f"Constraint operator must be '==', '<=', or '>=', got {op!r}.")


class Assemblage:
    r"""The operator-valued function \(K(a, b \mid x, y)\) of an extended nonlocal game as cvxpy expressions.

    For every question pair the block matrix \(K_{xy}\), whose \((a, b)\) block is \(K(a, b \mid x, y)\), is a linear
    image of one real variable vector: a Hermitian block has ``dim (dim + 1) / 2`` free real parts and
    ``dim (dim - 1) / 2`` free imaginary parts, and sparse matrices place them into the block matrix. Each block is
    therefore Hermitian by construction, and the block matrices, their slices and linear functionals of them all
    have small expression trees, which keeps cvxpy's canonicalisation cheap when there are hundreds of blocks.

    Without a referee (``dim == 1``) the blocks are real scalars and ``exprs[x, y]`` is a plain real variable of
    shape ``(num_alice_out, num_bob_out)``.
    """

    def __init__(self, referee_dim: int, alice_out: int, bob_out: int, alice_in: int, bob_in: int) -> None:
        """Create the variables and the block matrices of the assemblage."""
        self.shape = (referee_dim, referee_dim, alice_out, bob_out, alice_in, bob_in)
        dim = referee_dim
        self.exprs: dict[tuple[int, int], cvxpy.Expression] = {}
        if dim == 1:
            for x in range(alice_in):
                for y in range(bob_in):
                    self.exprs[x, y] = cvxpy.Variable((alice_out, bob_out), name=f"K({x},{y})")
            return

        num_blocks = alice_out * bob_out
        size = num_blocks * dim * dim
        real_entries, real_cols, imag_entries, imag_cols, imag_signs = [], [], [], [], []
        num_real = num_imag = 0
        for a in range(alice_out):
            for b in range(bob_out):
                for r in range(dim):
                    for s in range(r, dim):
                        upper = (a * dim + r) * (bob_out * dim) + (b * dim + s)
                        lower = (a * dim + s) * (bob_out * dim) + (b * dim + r)
                        real_entries += [upper, lower] if r != s else [upper]
                        real_cols += [num_real, num_real] if r != s else [num_real]
                        num_real += 1
                        if r != s:
                            imag_entries += [upper, lower]
                            imag_cols += [num_imag, num_imag]
                            imag_signs += [1.0, -1.0]
                            num_imag += 1
        self._real_map = scipy.sparse.csc_matrix(
            (np.ones(len(real_entries)), (real_entries, real_cols)), shape=(size, num_real)
        )
        self._imag_map = scipy.sparse.csc_matrix((imag_signs, (imag_entries, imag_cols)), shape=(size, num_imag))
        self._real = {(x, y): cvxpy.Variable(num_real) for x in range(alice_in) for y in range(bob_in)}
        self._imag = {(x, y): cvxpy.Variable(num_imag) for x in range(alice_in) for y in range(bob_in)}
        matrix_shape = (alice_out * dim, bob_out * dim)
        for key in self._real:
            self.exprs[key] = cvxpy.reshape(self._real_map @ self._real[key], matrix_shape, order="C") + 1j * (
                cvxpy.reshape(self._imag_map @ self._imag[key], matrix_shape, order="C")
            )

    def linear_functional(self, weights: np.ndarray, prob_mat: np.ndarray) -> cvxpy.Expression:
        r"""Return \(\sum_{x,y} \pi(x,y) \sum_{a,b} \langle W(a,b \mid x,y), K(a,b \mid x,y) \rangle\).

        The weights have the shape of the assemblage, and the inner product is the Hilbert-Schmidt one, so the
        result is real-valued. The coefficients of the variables are computed in numpy.
        """
        referee_dim, _, alice_out, bob_out, alice_in, bob_in = self.shape
        if referee_dim == 1:
            return sum(
                cvxpy.sum(cvxpy.multiply(prob_mat[x, y] * weights[0, 0, :, :, x, y].real, var))
                for (x, y), var in self.exprs.items()
            )
        total = 0
        for (x, y), real_var in self._real.items():
            block_matrix = prob_mat[x, y] * np.block(
                [[weights[:, :, a, b, x, y] for b in range(bob_out)] for a in range(alice_out)]
            )
            # <W, K> = sum conj(W) K, whose real part pairs Re(W) with the real parts and Im(W) with the imaginary.
            real_coeffs = self._real_map.T @ block_matrix.real.ravel()
            imag_coeffs = self._imag_map.T @ block_matrix.imag.ravel()
            total = total + real_coeffs @ real_var + imag_coeffs @ self._imag[x, y]
        return total

    def value(self) -> np.ndarray:
        """Return the current value of the assemblage as an array with the layout of a game's predicate."""
        referee_dim, _, alice_out, bob_out, alice_in, bob_in = self.shape
        optimum = np.zeros(self.shape, dtype=complex)
        for (x, y), expr in self.exprs.items():
            optimum[:, :, :, :, x, y] = expr.value.reshape(alice_out, referee_dim, bob_out, referee_dim).transpose(
                1, 3, 0, 2
            )
        return optimum


def npa_upper_bound(
    prob_mat: np.ndarray,
    pred_mat: np.ndarray,
    k: int | str,
    no_signaling: bool,
    constraints: list[AnswerEventConstraint] | None,
    solver: str | None,
    solver_params: dict | None,
) -> tuple[float, np.ndarray]:
    """Solve the level-``k`` NPA relaxation of the commuting measurement value.

    Returns the upper bound and the optimal assemblage of the relaxation as an array of shape
    ``(referee_dim, referee_dim, num_alice_out, num_bob_out, num_alice_in, num_bob_in)``.
    """
    if solver_params is None:
        solver_params = {"eps": 1e-8, "max_iters": 100_000} if solver == cvxpy.SCS else {}
    referee_dim, _, alice_out, bob_out, alice_in, bob_in = pred_mat.shape

    assemblage = Assemblage(referee_dim, alice_out, bob_out, alice_in, bob_in)
    # The winning probability is sum_{x, y} pi(x, y) sum_{a, b} <V(a, b | x, y), K(a, b | x, y)>.
    objective = assemblage.linear_functional(pred_mat, prob_mat)

    cons = npa_constraints(assemblage.exprs, k, referee_dim=referee_dim, no_signaling=no_signaling)
    for constraint in constraints or []:
        cons.append(
            answer_event_linear_constraint(
                assemblage.exprs, constraint, referee_dim, alice_out, bob_out, alice_in, bob_in
            )
        )
    value = solve_sdp(cvxpy.Problem(cvxpy.Maximize(objective), cons), solver, solver_params)
    return value, assemblage.value()
