"""Building blocks shared by nonlocal games and extended nonlocal games.

A nonlocal game is an extended nonlocal game whose referee has a one-dimensional system, so every routine here takes
the predicate in its six-dimensional form ``pred_mat[r, s, a, b, x, y]`` and a nonlocal game passes
``pred_mat[np.newaxis, np.newaxis]``.
"""

import itertools
from typing import Literal, TypeAlias

import cvxpy
import numpy as np

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
    # D_b^y = sum_{x, a} pi(x, y) Tr_R[(V(a, b | x, y) (x) 1) rho_a^x].
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

    # The assemblage K_xy is the block matrix of the referee's conditional states K_xy(a, b); without a referee it is
    # just the real matrix of answer probabilities.
    assemblage = {}
    for x in range(alice_in):
        for y in range(bob_in):
            if referee_dim == 1:
                assemblage[x, y] = cvxpy.Variable((alice_out, bob_out), name=f"K({x},{y})")
            else:
                blocks = [
                    [
                        cvxpy.Variable((referee_dim, referee_dim), hermitian=True, name=f"K({x},{y})_{a}{b}")
                        for b in range(bob_out)
                    ]
                    for a in range(alice_out)
                ]
                assemblage[x, y] = cvxpy.bmat(blocks)

    # The winning probability is sum_{x, y} <W_xy, K_xy> with W_xy the block matrix of weighted referee operators
    # pi(x, y) V(a, b | x, y), laid out in the same (a, b) block pattern as K_xy.
    total_win = 0
    for (x, y), var in assemblage.items():
        weights = prob_mat[x, y] * np.block(
            [[pred_mat[:, :, a, b, x, y] for b in range(bob_out)] for a in range(alice_out)]
        )
        total_win += cvxpy.sum(cvxpy.multiply(weights.conj(), var))

    cons = npa_constraints(assemblage, k, referee_dim=referee_dim, no_signaling=no_signaling)
    for constraint in constraints or []:
        cons.append(
            answer_event_linear_constraint(assemblage, constraint, referee_dim, alice_out, bob_out, alice_in, bob_in)
        )
    # Without a referee the objective is already real, and cvxpy's real() does not accept a real argument.
    objective = cvxpy.real(total_win) if total_win.is_complex() else total_win
    value = solve_sdp(cvxpy.Problem(cvxpy.Maximize(objective), cons), solver, solver_params)

    optimum = np.zeros((referee_dim, referee_dim, alice_out, bob_out, alice_in, bob_in), dtype=complex)
    for (x, y), var in assemblage.items():
        optimum[:, :, :, :, x, y] = var.value.reshape(alice_out, referee_dim, bob_out, referee_dim).transpose(
            1, 3, 0, 2
        )
    return value, optimum
