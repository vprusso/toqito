"""Weak coin flipping protocol."""
import numpy as np
from toqito.super_operators.partial_trace import partial_trace_cvx
import cvxpy


def weak_coin_flipping(rho: np.ndarray) -> float:
    """
    Weak coin flipping protocol.

    SDP from : https://arxiv.org/pdf/1703.03887.pdf
    """
    dims = rho.shape
    id_dim = int(np.sqrt(dims[0]))

    sdp_var = cvxpy.Variable(dims, PSD=True)
    objective = cvxpy.Maximize(cvxpy.trace(rho.conj().T * sdp_var))
    constraints = [
            partial_trace_cvx(sdp_var) == 1/id_dim * np.identity(id_dim)
            ]
    problem = cvxpy.Problem(objective, constraints)
    sol_default = problem.solve()

    return sol_default

