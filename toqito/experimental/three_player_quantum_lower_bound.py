"""Computes a lower bound on the quantum value of a nonlocal game."""
from typing import Dict, Tuple
import cvxpy
import numpy as np
from collections import defaultdict
from toqito.random.random_povm import random_povm


def optimize_alice(dim: int,
                   prob_mat: np.ndarray,
                   pred_mat: np.ndarray,
                   bob_povms,
                   charlie_povms) -> Tuple[Dict, float]:
    """
    Fix Bob's measurements and optimize over Alice and Charlie's measurements 
    in the semidefinite program.
    """
    # Get number of inputs and outputs.
    ia, ib, ic = prob_mat.shape
    oa, ob, oc = pred_mat.shape[0], pred_mat.shape[1], pred_mat.shape[2]

    # The cvxpy package does not support optimizing over 4-dimensional objects.
    # To overcome this, we use a dictionary to index between the questions and
    # answers, while the cvxpy variables held at this positions are
    # `dim`-by-`dim` cvxpy variables.
    alice_povms = defaultdict(cvxpy.Variable)
    for x in range(ia):
        for a in range(oa):
            alice_povms[x, a] = cvxpy.Variable((dim, dim), PSD=True)

#     charlie_povms = defaultdict(cvxpy.Variable)
#     for z in range(ic):
#         for c in range(oc):
#             charlie_povms[z, c] = cvxpy.Variable((dim, dim), PSD=True)

    tau = cvxpy.Variable((dim, dim), PSD=True)

    win = 0
    for x in range(ia):
        for y in range(ib):
            for z in range(ic):
                for a in range(oa):
                    for b in range(ob):
                        for c in range(oc):
                            win += prob_mat[x, y, z] * pred_mat[a, b, c, x, y, z] * \
                                   (cvxpy.trace(bob_povms[y, b].conj().T @ alice_povms[x, a]) + \
                                    cvxpy.trace(charlie_povms[z, c].conj() @ alice_povms[x, a]) + \
                                    cvxpy.trace(bob_povms[y, b].conj().T @ charlie_povms[z, c]))

    objective = cvxpy.Maximize(cvxpy.real(win))

    constraints = []

    # Sum over "a" for all "x" for Alice's measurements.
    for x in range(ia):
        alice_sum_a = 0
        for a in range(oa):
            alice_sum_a += alice_povms[x, a]
        constraints.append(alice_sum_a == tau)

#     for z in range(ic):
#         charlie_sum_c = 0
#         for c in range(oc):
#             charlie_sum_c += charlie_povms[z, c]
#         constraints.append(charlie_sum_c == tau)

    constraints.append(cvxpy.trace(tau) == 1)

    problem = cvxpy.Problem(objective, constraints)

    lower_bound = problem.solve()
    return alice_povms, lower_bound


def three_player_quantum_lower_bound(dim: int,
                                     prob_mat: np.ndarray,
                                     pred_mat: np.ndarray,
                                     iters: int = 5,
                                     tol: float = 10e-6,
                                     verbose: bool = True):
    # Get number of inputs and outputs.
    ia, ib, ic = prob_mat.shape
    oa, ob, oc = pred_mat.shape[0], pred_mat.shape[1], pred_mat.shape[2]

    best_lower_bound = float("-inf")
    for i in range(iters):
        if verbose:
            print(f"Starting iteration {i} of the alternating projections"
                  f" method.")
        # Generate a set of random POVMs for Bob. These measurements serve as a
        # rough starting point for the alternating projection algorithm.
        bob_tmp = random_povm(dim, ib, ob)
        bob_povms = defaultdict(int)
        for y in range(ib):
            for b in range(ob):
                bob_povms[y, b] = bob_tmp[:, :, y, b]

        charlie_tmp = random_povm(dim, ic, oc)
        charlie_povms = defaultdict(int)
        for z in range(ic):
            for c in range(oc):
                charlie_povms[z, c] = charlie_tmp[:, :, z, c]
 
        # Run the alternating projection algorithm between the two SDPs.
        it_diff = 1
        prev_win = -1
        best = float("-inf")
        
        alice_povms, lower_bound = optimize_alice(dim,
                                                  prob_mat,
                                                  pred_mat,
                                                  bob_povms,
                                                  charlie_povms)
        print(lower_bound)
#         while it_diff > tol:
#             # Optimize over Alice's measurement operators while fixing Bob's.
#             # If this is the first iteration, then the previously randomly
#             # generated operators in the outer loop are Bob's. Otherwise, Bob's
#             # operators come from running the next SDP.
#             alice_povm, lower_bound = optimize_alice(dim,
#                                                      prob_mat,
#                                                      pred_mat,
#                                                      bob_povm)
#             bob_povm, lower_bound = optimize_bob(dim,
#                                                  prob_mat,
#                                                  pred_mat,
#                                                  alice_povm)
# 
#             it_diff = lower_bound - prev_win
#             prev_win = lower_bound
# 
#             # As the SDPs keep alternating, check if the winning probability
#             # becomes any higher. If so, replace with new best.
#             if best < lower_bound:
#                 best = lower_bound
# 
#         if best_lower_bound < best:
#             if verbose:
#                 print(f"Best lower bound {best_lower_bound} for iteration {i}.")
#             best_lower_bound = best
# 
#     if verbose:
#         print(f"Best lower bound: {best_lower_bound} for a maximum of {iters}"
#               f" iterations.")
#     return best_lower_bound


def _optimize_alice(dim: int,
                   prob_mat: np.ndarray,
                   pred_mat: np.ndarray,
                   bob_povms) -> Tuple[Dict, float]:
    """
    Fix Bob's measurements and optimize over Alice's measurements in the
    semidefinite program.
    """
    # Get number of inputs and outputs.
    ia, ib = prob_mat.shape
    oa, ob = pred_mat.shape[0], pred_mat.shape[1]

    # The cvxpy package does not support optimizing over 4-dimensional objects.
    # To overcome this, we use a dictionary to index between the questions and
    # answers, while the cvxpy variables held at this positions are
    # `dim`-by-`dim` cvxpy variables.
    alice_povms = defaultdict(cvxpy.Variable)
    for x in range(ia):
        for a in range(oa):
            alice_povms[x, a] = cvxpy.Variable((dim, dim), PSD=True)

    tau = cvxpy.Variable((dim, dim), PSD=True)

    # ..math
    #    \sum_{(x,y) \in \Sigma} \pi(x, y) V(a,b|x,y) \ip{B_b^y}{A_a^x}
    win = 0
    is_real = True
    for x in range(ia):
        for y in range(ib):
            for a in range(oa):
                for b in range(ob):
                    if isinstance(bob_povms[y, b], np.ndarray):
                        win += prob_mat[x, y] * pred_mat[a, b, x, y] * \
                               cvxpy.trace(
                                   bob_povms[y, b].conj().T @ alice_povms[x, a]
                               )
                    if isinstance(bob_povms[y, b], cvxpy.expressions.variable.Variable):
                        is_real = False
                        win += prob_mat[x, y] * pred_mat[a, b, x, y] * \
                               cvxpy.trace(
                                   bob_povms[y, b].value.conj().T @ alice_povms[x, a]
                               )

    if is_real:
        objective = cvxpy.Maximize(cvxpy.real(win))
    else:
        objective = cvxpy.Maximize(win)

    constraints = []

    # Sum over "a" for all "x" for Alice's measurements.
    for x in range(ia):
        alice_sum_a = 0
        for a in range(oa):
            alice_sum_a += alice_povms[x, a]
        constraints.append(alice_sum_a == tau)

    constraints.append(cvxpy.trace(tau) == 1)

    problem = cvxpy.Problem(objective, constraints)

    lower_bound = problem.solve()
    return alice_povms, lower_bound


def optimize_bob(dim: int,
                 prob_mat: np.ndarray,
                 pred_mat: np.ndarray,
                 alice_povms) -> Tuple[Dict, float]:
    """
    Fix Alice's measurements and optimize over Bob's measurements in the
    semidefinite program.
    """
    # Get number of inputs and outputs.
    ia, ib = prob_mat.shape
    oa, ob = pred_mat.shape[0], pred_mat.shape[1]

    # Now, optimize over Bob's measurement operators and fix Alice's operators
    # as those are coming from the previous SDP.
    bob_povms = defaultdict(cvxpy.Variable)
    for y in range(ib):
        for b in range(ob):
            bob_povms[y, b] = cvxpy.Variable((dim, dim), PSD=True)

    win = 0
    for x in range(ia):
        for y in range(ib):
            for a in range(oa):
                for b in range(ob):
                    win += prob_mat[x, y] * pred_mat[a, b, x, y] * cvxpy.trace(
                        bob_povms[y, b].H @ alice_povms[x, a].value)

    objective = cvxpy.Maximize(win)
    constraints = []

    # Sum over "b" for all "y" for Bob's measurements.
    for y in range(ib):
        bob_sum_b = 0
        for b in range(ob):
            bob_sum_b += bob_povms[y, b]
        constraints.append(bob_sum_b == np.identity(dim))

    problem = cvxpy.Problem(objective, constraints)

    lower_bound = problem.solve()
    return bob_povms, lower_bound
