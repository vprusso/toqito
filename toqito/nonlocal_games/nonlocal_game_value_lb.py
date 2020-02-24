import cvxpy
import numpy as np
from collections import defaultdict
from toqito.random.random_density_matrix import random_density_matrix
from toqito.random.random_unitary import random_unitary
from toqito.random.random_povm import random_povm

from numpy import sin, cos, pi
from toqito.base.ket import ket
from toqito.matrix.properties.is_unitary import is_unitary


def nonlocal_game_value_lb(d, p, V, reps: int = 1, tol: float = 10e-6):

    # Get number of inputs and outputs.
    ia, ib = p.shape
    oa, ob = V.shape[0], V.shape[1]

    B = random_povm(d, ia, oa)

    A = defaultdict(int)
    for x in range(ia):
        for a in range(oa):
            #A[x, a] = cvxpy.Variable((d, d), hermitian=True)
            A[x, a] = cvxpy.Variable((d, d), PSD=True)

#    tau = cvxpy.Variable((d, d), hermitian=True)
    tau = cvxpy.Variable((d, d), PSD=True)

    win = 0
    for x in range(ia):
        for y in range(ib):
            for a in range(oa):
                for b in range(ob):
                    win += p[x, y] * V[a, b, x, y] * cvxpy.trace(B[:, :, y, b].conj().T @ A[x, a])

    objective = cvxpy.Maximize(cvxpy.real(win))
    constraints = []

    # Sum over "a" for all "x".
    for x in range(ia):
        s = 0
        for a in range(oa):
            s += A[x, a]
#            constraints.append(A[x, a] >> 0)
        constraints.append(s == tau)
#        constraints.append(s == np.eye(d))

    constraints.append(cvxpy.trace(tau) == 1)
#    constraints.append(tau >> 0)

    problem = cvxpy.Problem(objective, constraints)

    primal = problem.solve()
    print(primal)

