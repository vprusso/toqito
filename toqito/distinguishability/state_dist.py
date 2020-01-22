from typing import List
from toqito.super_operators.ptrace import partial_trace
from numpy import linalg as LA
import cvxpy as cvx
import numpy as np


def state_dist(states: List[np.ndarray],
               dims: List[int],
               axis: int,
               probs: List[float] = None):

    # Assume uniform probability if no specific distribution is given.
    if probs is None:
        probs = [1/len(states)] * len(states)

    if len(states) == 2:
        return 1/2 + 1/4*LA.norm(states[0] - states[1], 1)

    if len(states) >= 3:
        dim = states[0].shape
        
        obj_func = []
        sdp_vars = []
        constraints = []
        for i in range(len(states)):
            sdp_vars.append(cvx.Variable(dim, PSD=True))

            obj_func.append(probs[i] * cvx.trace(states[i].conj().T * sdp_vars[i]))

            constraints.append(partial_trace(sdp_vars[i],
                                             dims=dims,
                                             axis=axis) == np.identity(dim[0]//2))

        objective = cvx.Maximize(sum(obj_func))
        problem = cvx.Problem(objective, constraints)
        sol_default = problem.solve()
        return 1/len(states) * sol_default
