"""Semidefinite programs for obtaining values of quantum hedging scenarios."""
import cvxpy
import numpy as np

from toqito.perms.pi_perm import pi_perm
from toqito.super_operators.partial_trace import partial_trace_cvx


def max_prob_outcome_a_primal(q_a: np.ndarray, num_reps: int) -> float:
    """
    Compute the maximal probability for calculating outcome "a".

    :param q_a:
    :param num_reps:
    :return:
    """
    sys = list(range(1, 2*num_reps, 2))
    if len(sys) == 1:
        sys = sys[0]
    dim = 2 * np.ones((1, 2*num_reps)).astype(int).flatten()
    dim = dim.tolist()

    x_var = cvxpy.Variable((4**num_reps, 4**num_reps), PSD=True)
    objective = cvxpy.Maximize(cvxpy.trace(q_a.conj().T @ x_var))
    constraints = [
        partial_trace_cvx(x_var, sys, dim) == np.identity(2 ** num_reps)
    ]
    problem = cvxpy.Problem(objective, constraints)

    return problem.solve()


def max_prob_outcome_a_dual(q_a: np.ndarray, num_reps: int) -> float:
    """
    Compute the maximal probability for calculating outcome "a".

    :param q_a:
    :param num_reps:
    :return:
    """
    y_var = cvxpy.Variable((2**num_reps, 2**num_reps), hermitian=True)
    objective = cvxpy.Minimize(cvxpy.trace(cvxpy.real(y_var)))

    pperm = pi_perm(num_reps)
    kron_var = cvxpy.kron(np.eye(2**num_reps), y_var)
    pperm_conj = pi_perm(num_reps).conj().T

    if num_reps == 1:
        u_var = cvxpy.multiply(cvxpy.multiply(pperm, kron_var), pperm_conj)
        constraints = [cvxpy.real(u_var) >> q_a]
    else:
        constraints = [cvxpy.real(pperm @ kron_var @ pperm_conj) >> q_a]
    problem = cvxpy.Problem(objective, constraints)

    return problem.solve()


def min_prob_outcome_a_primal(q_a: np.ndarray, num_reps: int) -> float:
    """
    Compute the minimal probability for calculating outcome "a".

    :param q_a:
    :param num_reps:
    :return:
    """
    sys = list(range(1, 2*num_reps, 2))
    if len(sys) == 1:
        sys = sys[0]
    dim = 2 * np.ones((1, 2*num_reps)).astype(int).flatten()
    dim = dim.tolist()

    x_var = cvxpy.Variable((4**num_reps, 4**num_reps), PSD=True)
    objective = cvxpy.Minimize(cvxpy.trace(q_a.conj().T @ x_var))
    constraints = [
        partial_trace_cvx(x_var, sys, dim) == np.identity(2 ** num_reps)
    ]
    problem = cvxpy.Problem(objective, constraints)

    return problem.solve()


def min_prob_outcome_a_dual(q_a: np.ndarray, num_reps: int) -> float:
    """
    Compute the minimal probability for calculating outcome "a".

    :param q_a:
    :param num_reps:
    :return:
    """
    y_var = cvxpy.Variable((2**num_reps, 2**num_reps), hermitian=True)
    objective = cvxpy.Maximize(cvxpy.trace(cvxpy.real(y_var)))

    pperm = pi_perm(num_reps)
    kron_var = cvxpy.kron(np.eye(2**num_reps), y_var)
    pperm_conj = pi_perm(num_reps).conj().T

    if num_reps == 1:
        u_var = cvxpy.multiply(cvxpy.multiply(pperm, kron_var), pperm_conj)
        constraints = [cvxpy.real(u_var) << q_a]
    else:
        constraints = [cvxpy.real(pperm @ kron_var @ pperm_conj) << q_a]
    problem = cvxpy.Problem(objective, constraints)

    return problem.solve()
