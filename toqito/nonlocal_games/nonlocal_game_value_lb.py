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

    # Generate a set of random POVMs for Bob. These measurements serve as a
    # rough starting point for the alternating projection algorithm.
    B_tmp = random_povm(d, ia, oa)
    B = defaultdict(int)
    for y in range(ib):
        for b in range(ob):
            B[y, b] = B_tmp[:, :, y, b]

    # Run the alternating projection algorithm between the two SDPs.
    it_diff = 1
    prev_win = -1
#    while it_diff > tol:
        # Optimize over Alice's measurement operators while fixing Bob's. If
        # this is the first iteration, then the previously randomly generated
        # operators in the outer loop are Bob's. Otherwise, Bob's operators
        # come from running the next SDP.

    # The cvxpy package does not support optimizing over 4-dimensional objects.
    # To overcome this, we use a dictionary to index between the questions and
    # answers, while the cvxpy variables held at this positions are
    # `dim`-by-`dim` cvxpy variables.
    A = defaultdict(int)
    for x in range(ia):
        for a in range(oa):
            A[x, a] = cvxpy.Variable((d, d), PSD=True)

    tau = cvxpy.Variable((d, d), PSD=True)

    # ..math
    #    \sum_{(x,y) \in \Sigma} \pi(x, y) V(a,b|x,y) \ip{B_b^y}{A_a^x}
    win = 0
    for x in range(ia):
        for y in range(ib):
            for a in range(oa):
                for b in range(ob):
                    win += p[x, y] * V[a, b, x, y] * cvxpy.trace(B[y, b].conj().T @ A[x, a])
    
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

    # Now, optimize over Bob's measurement operators and fix Alice's operators
    # as those are coming from the previous SDP.
    B = defaultdict(int)
    for y in range(ib):
        for b in range(ob):
            B[y, b] = cvxpy.Variable((d, d), PSD=True)

    win = 0
    for x in range(ia):
        for y in range(ib):
            for a in range(oa):
                for b in range(ob):
                    win += p[x, y] * V[a, b, x, y] * cvxpy.trace(B[y, b] @ A[x, a].value)

    objective = cvxpy.Maximize(win)
    constraints = []
    for y in range(ib):
        s = 0
        for b in range(ob):
            s += B[y, b]
        constraints.append(s == np.identity(d))

    problem = cvxpy.Problem(objective, constraints)

    primal = problem.solve()
    print(primal)


