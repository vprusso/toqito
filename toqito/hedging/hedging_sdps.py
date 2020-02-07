import cvxpy
import numpy as np

from toqito.super_operators.partial_trace import partial_trace_cvx
from toqito.hedging.pi_perm import pi_perm


def maximize_losing_less_than_k(Q_a, n):
    sys = list(range(1, 2**(n-1), 2))
    if len(sys) == 1:
        sys = sys[0]

    dim = 2*np.ones((1, 2**(n-1))).astype(int).flatten()
    dim = dim.tolist()

    X = cvxpy.Variable((4**(n-1), 4**(n-1)), PSD=True)
    objective = cvxpy.Maximize(cvxpy.trace(Q_a.conj().T @ X))
    constraints = [partial_trace_cvx(X, sys, dim) == np.identity(2**(n-1))]
    problem = cvxpy.Problem(objective, constraints)

    primal = problem.solve()
    print(primal)

    Y = cvxpy.Variable((2**(n-1), 2**(n-1)), hermitian=True)
    objective = cvxpy.Minimize(cvxpy.trace(cvxpy.real(Y)))

    a = pi_perm(n-1)
    b = cvxpy.kron(np.identity(2**(n-1)), Y)
    c = pi_perm(n-1).conj().T
    u = cvxpy.multiply(cvxpy.multiply(a, b), c)

    constraints = [cvxpy.real(u) >= Q_a]
    problem = cvxpy.Problem(objective, constraints)

    dual = problem.solve()
    print(dual)
    

