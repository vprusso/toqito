"""Weak coin flipping protocol."""
import cvxpy
import numpy as np
from toqito.channels.channels.partial_trace import partial_trace_cvx


def weak_coin_flipping(rho: np.ndarray) -> float:
    """
    Weak coin flipping protocol [MS]_.

    Examples
    ==========

    References
    ==========
    .. [MS] Ganz, Maor and Sattath, Or
        Quantum coin hedging, and a counter measure
        https://arxiv.org/pdf/1703.03887.pdf
    """
    dims = rho.shape
    id_dim = int(np.sqrt(dims[0]))

    sdp_var = cvxpy.Variable(dims, PSD=True)
    objective = cvxpy.Maximize(cvxpy.trace(rho.conj().T @ sdp_var))
    constraints = [partial_trace_cvx(sdp_var) == 1 / id_dim * np.identity(id_dim)]
    problem = cvxpy.Problem(objective, constraints)
    sol_default = problem.solve()

    return sol_default
