import cvxpy
import numpy as np

#from toqito.super_operators.partial_trace import partial_trace_cvx
from toqito.base.ket import ket
from toqito.super_operators.dephasing_channel import dephasing_channel


def diamond_norm() -> float:
    """
    """

    e0, e1 = ket(2, 0), ket(2, 1)
    e00 = np.kron(e0, e0)
    e11 = np.kron(e1, e1)

    u = 1/np.sqrt(2) * (e00 + e11)
    rho = u * u.conj().T
    Y = cvxpy.Variable((4, 4))
    rho0 = cvxpy.Variable((4, 4), PSD=True)
    rho1 = cvxpy.Variable((4, 4), PSD=True)
    objective = cvxpy.Maximize(1/2 * cvxpy.trace(Y) + 1/2 * cvxpy.trace(Y.H))
    dephasing_channel(rho0)
#    constraints = [cvxpy.bmat([[dephasing_channel(rho0), Y], [Y.H, dephasing_channel(rho1)]]),
#                   cvxpy.trace(rho0) == 1,
#                   cvxpy.trace(rho1) == 1]
    #problem = cvxpy.Problem(objective, constraints)

    #primal = problem.solve()
    #print(primal)

#    objective = cvxpy.Minimize(cvxpy.norm(partial_trace_cvx(y0_var, 2, dim)) + cvxpy.norm(partial_trace_cvx(y1_var, 2, dim)))

    # X = cvxpy.Variable((4**n, 4**n), PSD=True)
    # objective = cvxpy.Maximize(cvxpy.trace(Q_a.conj().T @ X))
    # constraints = [partial_trace_cvx(X, sys, dim) == np.identity(2**n)]
    # problem = cvxpy.Problem(objective, constraints)
    #
    # primal = problem.solve()
    # print(primal)


