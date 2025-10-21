"""Evaluate the quantum learnability semidefinite programs."""

import warnings
from itertools import combinations
from typing import Any, Iterable, Sequence

import cvxpy as cp
import cvxpy.settings as cp_settings
import numpy as np
import scipy.sparse as sp

from toqito.matrix_ops import to_density_matrix, vectors_to_gram_matrix
from toqito.matrix_props import is_positive_semidefinite, is_rank_one


def learnability(
    states: Sequence[np.ndarray],
    k: int,
    *,
    solver: str | None = "SCS",
    solver_kwargs: dict[str, Any] | None = None,
    verify_reduced: bool = True,
    verify_tolerance: float = 1e-4,
    tol: float = 1e-8,
) -> dict[str, float | str | None]:
    r"""Compute the average error value of the learnability semidefinite program.

    This routine minimizes

    .. math::

        \frac{1}{n} \sum_{i = 1}^n \left\langle \rho_i,
        \sum_{S: i \notin S} M_S \right\rangle.

    over POVM elements :math:`(M_S)` indexed by ``k``-element subsets, subject to
    :math:`\sum_S M_S = \mathbb{I}` and :math:`M_S \succeq 0`.  When all inputs are pure, the
    reduced Gram-matrix SDP

    .. math::

        \sum_{i = 1}^n \bra{i} \sum_{S: i \notin S} W_S \ket{i}.

    with constraint :math:`\sum_S W_S = G` (Gram matrix) and :math:`W_S \succeq 0`
    is also solved as a consistency check.

    Examples
    ========

    .. jupyter-execute::

        from toqito.state_props import learnability
        from toqito.states import basis

        e0, e1 = basis(2, 0), basis(2, 1)
        learnability(
            [e0, e1],
            k=1,
            solver="SCS",
            solver_kwargs={"eps": 1e-6, "max_iters": 5_000},
        )

    :param states: Sequence of state vectors or density matrices acting on the same space.
    :param k: Subset size for the POVM outcomes; must satisfy :code:`1 <= k <= len(states)`.
    :param solver: Optional CVXPY solver name. Defaults to :code:`"SCS"`.
    :param solver_kwargs: Extra keyword arguments forwarded to :meth:`cvxpy.Problem.solve`.
    :param verify_reduced: If :code:`True` and the states are pure, also solve the reduced SDP.
    :param verify_tolerance: Absolute tolerance used when comparing the two optimal values.
    :param tol: Numerical tolerance used when validating positivity and rank-one states.
    :return: Dictionary with keys :code:`value`, :code:`total_value`, :code:`status`,
        :code:`measurement_operators`, and optionally :code:`reduced_value`,
        :code:`reduced_total_value`, :code:`reduced_status`, :code:`reduced_operators`.
    :raises ValueError: If the data are inconsistent with valid quantum states or if :code:`k`
        lies outside the permissible range.
    :raises cvxpy.error.SolverError: If the selected solver reports a failure.

    """
    if not states:
        raise ValueError("The list of states must be non-empty.")

    density_matrices, candidate_vectors = _convert_states(states, tol=tol)
    general_value, general_status, measurement_variables = _solve_learnability_general(
        density_matrices,
        k,
        solver=solver,
        solver_kwargs=solver_kwargs,
    )

    operator_values = {
        combo: measurement_variables[combo].value
        for combo in measurement_variables
    }

    result: dict[str, float | str | None | dict] = {
        "value": float(np.real(general_value)),
        "status": general_status,
        "reduced_value": None,
        "reduced_status": None,
        "measurement_operators": operator_values,
        "reduced_operators": None,
        "total_value": float(np.real(general_value)) * len(density_matrices),
    }
    result["reduced_total_value"] = None

    if verify_reduced and candidate_vectors is not None:
        gram = vectors_to_gram_matrix(candidate_vectors)
        reduced_value, reduced_status, reduced_variables = _solve_learnability_reduced(
            gram,
            k,
            solver=solver,
            solver_kwargs=solver_kwargs,
        )
        reduced_operator_values = {
            combo: var.value for combo, var in reduced_variables.items()
        }
        result["reduced_value"] = float(np.real(reduced_value))
        result["reduced_status"] = reduced_status
        result["reduced_operators"] = reduced_operator_values
        result["reduced_total_value"] = float(np.real(reduced_value)) * len(density_matrices)

        if abs(result["value"] - result["reduced_value"]) > verify_tolerance:
            warnings.warn(
                (
                    "General and reduced SDP optimal values differ by more than "
                    f"{verify_tolerance}. General value: {result['value']}, "
                    f"reduced value: {result['reduced_value']}."
                ),
                RuntimeWarning,
            )

    return result


def _solve_learnability_general(
    density_matrices: Sequence[np.ndarray],
    k: int,
    *,
    solver: str | None,
    solver_kwargs: dict[str, Any] | None,
) -> tuple[float, str, dict[tuple[int, ...], cp.Variable]]:
    n = len(density_matrices)
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= n (= {n}).")

    dim = density_matrices[0].shape[0]
    combos = list(combinations(range(n), k))
    variables = {
        combo: cp.Variable((dim, dim), hermitian=True)
        for combo in combos
    }

    constraints = [var >> 0 for var in variables.values()]
    constraints.append(_sum_expressions(variables.values()) == np.eye(dim, dtype=np.complex128))

    objective_terms = []
    for idx, rho in enumerate(density_matrices):
        without_idx = [var for combo, var in variables.items() if idx not in combo]
        if not without_idx:
            objective_terms.append(0.0)
            continue
        summed = _sum_expressions(without_idx)
        objective_terms.append(cp.real(cp.trace(rho @ summed)) / n)

    problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), constraints)
    value, status = _solve_problem(problem, solver, solver_kwargs)

    return value, status, variables


def _solve_learnability_reduced(
    gram_matrix: np.ndarray,
    k: int,
    *,
    solver: str | None,
    solver_kwargs: dict[str, Any] | None,
) -> tuple[float, str, dict[tuple[int, ...], cp.Variable]]:
    n = gram_matrix.shape[0]
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= n (= {n}).")

    combos = list(combinations(range(n), k))
    variables = {
        combo: cp.Variable((n, n), hermitian=True)
        for combo in combos
    }

    constraints = [var >> 0 for var in variables.values()]
    constraints.append(_sum_expressions(variables.values()) == gram_matrix)

    objective_terms = []
    for idx in range(n):
        without_idx = [var for combo, var in variables.items() if idx not in combo]
        if not without_idx:
            objective_terms.append(0.0)
            continue
        summed = _sum_expressions(without_idx)
        objective_terms.append(cp.real(summed[idx, idx]) / n)

    problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), constraints)
    value, status = _solve_problem(problem, solver, solver_kwargs)

    return value, status, variables


def _convert_states(
    states: Sequence[np.ndarray],
    *,
    tol: float,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    """Normalize input states and detect whether they are uniformly pure.

    Each entry in :code:`states` may be a state vector or a density matrix. The
    routine converts every element to a unit-trace density matrix, checks
    positivity, and records the original pure state vectors when all inputs are
    rank one.

    :param states: Collection of quantum states to normalize.
    :param tol: Numerical tolerance used for positivity and rank checks.
    :return: List of density matrices and, when available, the corresponding
        state vectors.

    """
    density_matrices: list[np.ndarray] = []
    pure_vectors: list[np.ndarray] = []
    all_pure = True
    dim: int | None = None

    for raw_state in states:
        state_array = np.asarray(raw_state, dtype=np.complex128)
        rho = to_density_matrix(state_array)
        rho = (rho + rho.conj().T) / 2

        trace = np.trace(rho)
        if np.isclose(trace, 0.0, atol=tol):
            raise ValueError("Each state must have strictly positive trace.")
        rho = rho / trace

        if dim is None:
            dim = rho.shape[0]
        elif rho.shape != (dim, dim):
            raise ValueError("All states must act on the same Hilbert space.")

        if not is_positive_semidefinite(rho, atol=tol):
            raise ValueError("Each state must be positive semidefinite.")

        density_matrices.append(rho)

        if all_pure:
            if is_rank_one(rho, tol=tol):
                pure_vectors.append(_extract_state_vector(state_array, rho))
            else:
                all_pure = False

    if not all_pure:
        pure_vectors = None

    return density_matrices, pure_vectors


def _solve_problem(
    problem: cp.Problem,
    solver: str | None,
    solver_kwargs: dict[str, Any] | None,
) -> tuple[float, str]:
    """Solve a CVXPY problem and return both the optimal value and status."""
    solve_kwargs = dict(solver_kwargs or {})

    if _is_scs_solver(solver):
        return _solve_problem_with_scs(problem, solve_kwargs)

    if solver is None:
        value = problem.solve(**solve_kwargs)
    else:
        value = problem.solve(solver=solver, **solve_kwargs)
    return value, problem.status


def _solve_problem_with_scs(
    problem: cp.Problem,
    solver_kwargs: dict[str, Any],
) -> tuple[float, str]:
    """Solve with SCS ensuring sparse matrices use CSC format to avoid warnings."""
    warm_start = bool(solver_kwargs.pop("warm_start", False))
    verbose = bool(solver_kwargs.pop("verbose", False))

    data, chain, inverse_data = problem.get_problem_data(cp.SCS)
    for key in (cp_settings.A, cp_settings.P):
        if key in data and data[key] is not None:
            data[key] = sp.csc_matrix(data[key])

    solution = chain.solve_via_data(
        problem,
        data,
        warm_start=warm_start,
        verbose=verbose,
        solver_opts=solver_kwargs,
    )
    problem.unpack_results(solution, chain, inverse_data)
    return problem.value, problem.status


def _is_scs_solver(solver: Any | None) -> bool:
    """Return True when the requested solver corresponds to SCS."""
    if solver is None:
        return False
    if solver is cp.SCS:
        return True
    if isinstance(solver, str) and solver.strip().upper() == "SCS":
        return True
    return False


def _extract_state_vector(
    original: np.ndarray,
    density: np.ndarray,
) -> np.ndarray:
    if original.ndim == 1:
        vector = original
    elif original.ndim == 2 and 1 in original.shape:
        vector = original.reshape(-1)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(density)
        vector = eigenvectors[:, np.argmax(eigenvalues)]

    norm = np.linalg.norm(vector)
    return (vector / norm).astype(np.complex128)


def _sum_expressions(expressions: Iterable[cp.expressions.expression.Expression]):
    iterator = iter(expressions)
    try:
        total = next(iterator)
    except StopIteration:
        return 0.0
    for expr in iterator:
        total = total + expr
    return total
