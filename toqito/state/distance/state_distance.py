"""Distinguish a set of quantum states."""
from typing import List
import cvxpy as cvx
import numpy as np
from toqito.super_operators.partial_trace import partial_trace_cvx


def state_distance(states: List[np.ndarray],
                   sys: List[int],
                   dims: List[int],
                   probs: List[float] = None) -> float:
    """
    Calculate the probability that a given quantum state can be distinguished
    from a set of quantum states.

    :param states:
    :param sys:
    :param dims:
    :param probs:
    :return:
    """

    # Assume uniform probability if no specific distribution is given.
    if probs is None:
        probs = [1/len(states)] * len(states)

    if len(states) == 2:
        return 1/2 + 1/4*np.linalg.norm(states[0] - states[1], 1)

    dim = states[0].shape

    obj_func = []
    sdp_vars = []
    constraints = []
    for i, _ in enumerate(states):
        sdp_vars.append(cvx.Variable(dim, PSD=True))

        obj_func.append(probs[i] * cvx.trace(states[i].conj().T * sdp_vars[i]))

        constraints.append(partial_trace_cvx(
            sdp_vars[i],
            sys=sys,
            dim=dims) == np.identity(dim[0]//2))

    objective = cvx.Maximize(sum(obj_func))
    problem = cvx.Problem(objective, constraints)
    sol_default = problem.solve()

    return 1/len(states) * sol_default
