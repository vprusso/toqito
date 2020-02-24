import cvxpy
import numpy as np
from collections import defaultdict
from toqito.random.random_povm import random_povm


def opt_alice(d, p, V, B):
    # Get number of inputs and outputs.
    ia, ib = p.shape
    oa, ob = V.shape[0], V.shape[1]

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
    f = True
    for x in range(ia):
        for y in range(ib):
            for a in range(oa):
                for b in range(ob):
                    if isinstance(B[y, b], np.ndarray):
                        win += p[x, y] * V[a, b, x, y] * cvxpy.trace(B[y, b].conj().T @ A[x, a])
                    if isinstance(B[y, b], cvxpy.expressions.variable.Variable):
                        f = False
                        #print(B[y, b].value.conj().T @ A[x, a])
                        win += p[x, y] * V[a, b, x, y] * cvxpy.trace(B[y, b].value.conj().T @ A[x, a])
   
    if not f:
        objective = cvxpy.Maximize(win)
    else:
        objective = cvxpy.Maximize(cvxpy.real(win))
    constraints = []

    # Sum over "a" for all "x".
    for x in range(ia):
        s = 0
        for a in range(oa):
            s += A[x, a]
        constraints.append(s == tau)

    constraints.append(cvxpy.trace(tau) == 1)

    problem = cvxpy.Problem(objective, constraints)

    win_value = problem.solve()
    return A, win_value


def opt_bob(d, p, V, A):
    # Get number of inputs and outputs.
    ia, ib = p.shape
    oa, ob = V.shape[0], V.shape[1]
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
                    win += p[x, y] * V[a, b, x, y] * cvxpy.trace(B[y, b].H @ A[x, a].value)

    objective = cvxpy.Maximize(win)
    constraints = []
    for y in range(ib):
        s = 0
        for b in range(ob):
            s += B[y, b]
        constraints.append(s == np.identity(d))

    problem = cvxpy.Problem(objective, constraints)

    win_value = problem.solve()
    return B, win_value


def nonlocal_game_value_lb(d, p, V, iters: int = 5, tol: float = 10e-6):

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

    best_lb = float("-inf")
    for i in range(iters):
        # Run the alternating projection algorithm between the two SDPs.
        it_diff = 1
        prev_win = -1
        best = float("-inf")
        while it_diff > tol:
            # Optimize over Alice's measurement operators while fixing Bob's. If
            # this is the first iteration, then the previously randomly generated
            # operators in the outer loop are Bob's. Otherwise, Bob's operators
            # come from running the next SDP.
            A, win_value = opt_alice(d, p, V, B)
            B, win_value = opt_bob(d, p, V, A)
            it_diff = win_value - prev_win
            prev_win = win_value
            print(win_value)
            if best < win_value:
                best = win_value
        if best_lb < best:
            best_lb = best
            print(best_lb)



