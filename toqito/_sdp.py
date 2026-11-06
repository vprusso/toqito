"""Shared helper for solving cvxpy problems with a status check."""

import cvxpy


def solve_sdp(problem: cvxpy.Problem, solver: str | None = None, solver_params: dict | None = None) -> float:
    """Solve a cvxpy problem and return its value, raising if the solver did not reach an optimum.

    Args:
        problem: The problem to solve.
        solver: The cvxpy solver to use (default is cvxpy's choice).
        solver_params: Extra keyword arguments passed to `cvxpy.Problem.solve`.

    Raises:
        ValueError: If the status after solving is neither optimal nor optimal-inaccurate, or no value was produced.

    Returns:
        The optimal value.

    """
    problem.solve(solver=solver, **({} if solver_params is None else solver_params))
    if problem.status not in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE) or problem.value is None:
        raise ValueError(f"The semidefinite program did not solve successfully (status: {problem.status}).")
    return float(problem.value)
