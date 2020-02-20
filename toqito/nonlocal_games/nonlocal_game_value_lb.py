import cvxpy
import numpy as np
from collections import defaultdict
from toqito.random.random_density_matrix import random_density_matrix


def nonlocal_game_value_lb(d, p, V):

    # Get some basic values.
    ma, mb = p.shape
    oa, ob = V.shape[0], V.shape[1]

    # Generate random starting measurements for Bob.
    B = np.zeros([d, d, ob+1, mb], order="F", dtype=complex)
    for y in range(mb):
        sum_B = np.zeros(d)
        for b in range(ob):
            B[:, :, b, y] = random_density_matrix(d, False, 1)
            sum_B = np.add(sum_B, B[:, :, b, y])
        lam = np.linalg.norm(sum_B)
        B[:, :, :, y] = B[:, :, :, y] / (lam + 0.1)
        B[:, :, ob, y] = np.eye(d) - sum_B / (lam + 0.1)

    B_meas = defaultdict(int)
    for i in range(ob):
        for j in range(mb):
            B_meas[i, j] = B[:, :, i, j]

    # Now loop until convergence is reached.
    it_diff = 1
    nglb = -1
    ct = 0
    
    q = defaultdict(int)
    for i in range(oa):
        for j in range(ob):
            q[i, j] = cvxpy.Variable((ma, mb))

    rho = cvxpy.Variable((d, d), hermitian=True)

    A = defaultdict(int)
    for i in range(d):
        for j in range(d):
            A[i, j] = cvxpy.Variable((oa, ma), hermitian=True)

    win = 0
    for x in range(ma):
        for y in range(mb):
            for a in range(oa):
                for b in range(ob):
                    for i in range(d):
                        for j in range(d):
                            win = win + cvxpy.trace(B_meas[i, j].conj().T @ q[i, j])
    objective = cvxpy.Maximize(cvxpy.real(win))

    constraints = []
    constraints.append(rho >> 0)
    constraints.append(cvxpy.trace(rho) == 1)

    tmp = []
    for a in reversed(list(range(oa))):
        for b in reversed(list(range(ob))):
            for x in reversed(list(range(ma))):
                for y in reversed(list(range(mb))):
                    for i in range(d):
                        for j in range(d):
                            tmp.append(q[i, j] == cvxpy.trace(B_meas[i, j].conj().T @ A[i, j]))
    constraints.extend(tmp)

    problem = cvxpy.Problem(objective, constraints)
    print(problem.solve())

#                    win = win + cvxpy.trace()
    
    # X = cvxpy.Variable((4**n, 4**n), PSD=True)
    # objective = cvxpy.Maximize(cvxpy.trace(Q_a.conj().T @ X))
    # constraints = [partial_trace_cvx(X, sys, dim) == np.identity(2**n)]
    # problem = cvxpy.Problem(objective, constraints)
    #
    # primal = problem.solve()
    # print(primal)
    
#    while it_diff > 10**-6:

