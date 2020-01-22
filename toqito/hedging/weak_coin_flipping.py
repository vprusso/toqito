import numpy as np
from toqito.helper.constants import e0, e1, e00, e01, e10, e11, em, ep
from toqito.states.bell import bell
from toqito.super_operators.ptrace import partial_trace
import cvxpy as cvx

def weak_coin_flipping():
    """
    SDP from : https://arxiv.org/pdf/1703.03887.pdf
    """
    rho0_AM = bell(0) * bell(0).conj().T

    state = np.kron(e1*e1.conj().T, e0*e0.conj().T) + np.kron(em*em.conj().T, e1*e1.conj().T)

    rho1_AM = cvx.Variable((4, 4), PSD=True)
    objective = cvx.Maximize(cvx.trace(state * rho1_AM))
    constraints = [partial_trace(rho1_AM, dims=[2, 2], axis=1) == 1/2 * np.identity(2)]
    problem = cvx.Problem(objective, constraints)
    sol_default = problem.solve()

    print(sol_default)
    print(np.around(rho1_AM.value, decimals=4))
