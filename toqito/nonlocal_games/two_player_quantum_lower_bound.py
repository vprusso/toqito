"""Computes a lower bound on the quantum value of a nonlocal game."""
from typing import Dict, Tuple
from collections import defaultdict
import cvxpy
import numpy as np
from toqito.random.random_povm import random_povm


def two_player_quantum_lower_bound(dim: int,
                                   prob_mat: np.ndarray,
                                   pred_mat: np.ndarray,
                                   iters: int = 5,
                                   tol: float = 10e-6):
    """
    Compute a lower bound on the quantum value of a nonlocal game.

    Calculates a lower bound on the maximum value that the specified nonlocal
    game can take on in quantum mechanical settings where Alice and Bob each
    have access to `d`-dimensional quantum system.

    This function works by starting with a randomly-generated POVM for Bob, and
    then optimizing Alice's POVM and the shared entangled state. Then Alice's
    POVM and the entangled state are fixed, and Bob's POVM is optimized. And so
    on, back and forth between Alice and Bob until convergence is reached.

    Note that the algorithm is not guaranteed to obtain the optimal local bound
    and can get stuck in local minimum values. The alleviate this, the `iter`
    parameter allows one to run the algorithm some pre-specified number of
    times and keep the highest value obtained.

    References:
    [1] Liang, Yeong-Cherng, and Andrew C. Doherty.
        "Bounds on quantum correlations in Bell-inequality experiments."
        Physical Review A 75.4 (2007): 042103.
        https://arxiv.org/abs/quant-ph/0608128

    :param dim: The local dimension (e.g. `dim = 2` corresponds to Alice and
                Bob each having access to a qubit.
    :param prob_mat: A matrix whose (x,y)-entry is the probability that the
                     referee asks Alice question x and Bob question y.
    :param pred_mat: A 4-dimensional array whose (a,b,x,y)-entry is the value
                    given to Alice and Bob when they provide answers a and b,
                    respectively, to questions x and y.
    :param iters: The number of times to run the alternating projection
                  algorithm.
    :param tol: The tolerance before quitting out of the alternating projection
                semidefinite program.
    :return: The lower bound on the quantum value of a nonlocal game.
    """
    # Get number of inputs and outputs.
    num_inputs_bob = prob_mat.shape[1]
    num_outputs_bob = pred_mat.shape[1]

    best_lower_bound = float("-inf")
    for _ in range(iters):
        # Generate a set of random POVMs for Bob. These measurements serve as a
        # rough starting point for the alternating projection algorithm.
        bob_tmp = random_povm(dim, num_inputs_bob, num_outputs_bob)
        bob_povms = defaultdict(int)
        for y_ques in range(num_inputs_bob):
            for b_ans in range(num_outputs_bob):
                bob_povms[y_ques, b_ans] = bob_tmp[:, :, y_ques, b_ans]

        # Run the alternating projection algorithm between the two SDPs.
        it_diff = 1
        prev_win = -1
        best = float("-inf")
        while it_diff > tol:
            # Optimize over Alice's measurement operators while fixing Bob's.
            # If this is the first iteration, then the previously randomly
            # generated operators in the outer loop are Bob's. Otherwise, Bob's
            # operators come from running the next SDP.
            alice_povms, lower_bound = optimize_alice(dim,
                                                      prob_mat,
                                                      pred_mat,
                                                      bob_povms)
            bob_povms, lower_bound = optimize_bob(dim,
                                                  prob_mat,
                                                  pred_mat,
                                                  alice_povms)

            it_diff = lower_bound - prev_win
            prev_win = lower_bound

            # As the SDPs keep alternating, check if the winning probability
            # becomes any higher. If so, replace with new best.
            if best < lower_bound:
                best = lower_bound

        if best_lower_bound < best:
            best_lower_bound = best

    return best_lower_bound


def optimize_alice(dim: int,
                   prob_mat: np.ndarray,
                   pred_mat: np.ndarray,
                   bob_povms) -> Tuple[Dict, float]:
    """Fix Bob's measurements and optimize over Alice's measurements."""
    # Get number of inputs and outputs.
    num_inputs_alice, num_inputs_bob = prob_mat.shape
    num_outputs_alice, num_outputs_bob = pred_mat.shape[0], pred_mat.shape[1]

    # The cvxpy package does not support optimizing over 4-dimensional objects.
    # To overcome this, we use a dictionary to index between the questions and
    # answers, while the cvxpy variables held at this positions are
    # `dim`-by-`dim` cvxpy variables.
    alice_povms = defaultdict(cvxpy.Variable)
    for x_ques in range(num_inputs_alice):
        for a_ans in range(num_outputs_bob):
            alice_povms[x_ques, a_ans] = cvxpy.Variable((dim, dim), PSD=True)

    tau = cvxpy.Variable((dim, dim), PSD=True)

    # ..math
    #    \sum_{(x,y) \in \Sigma} \pi(x, y) V(a,b|x,y) \ip{B_b^y}{A_a^x}
    win = 0
    is_real = True
    for x_ques in range(num_inputs_alice):
        for y_ques in range(num_inputs_bob):
            for a_ans in range(num_outputs_alice):
                for b_ans in range(num_outputs_bob):
                    if isinstance(bob_povms[y_ques, b_ans], np.ndarray):
                        win += prob_mat[x_ques, y_ques] * \
                               pred_mat[a_ans, b_ans, x_ques, y_ques] * \
                               cvxpy.trace(bob_povms[y_ques, b_ans].conj().T @
                                           alice_povms[x_ques, a_ans])
                    if isinstance(bob_povms[y_ques, b_ans],
                                  cvxpy.expressions.variable.Variable):
                        is_real = False
                        win += prob_mat[x_ques, y_ques] * \
                            pred_mat[a_ans, b_ans, x_ques, y_ques] * \
                            cvxpy.trace(
                                bob_povms[y_ques, b_ans].value.conj().T @
                                alice_povms[x_ques, a_ans]
                            )

    if is_real:
        objective = cvxpy.Maximize(cvxpy.real(win))
    else:
        objective = cvxpy.Maximize(win)

    constraints = []

    # Sum over "a" for all "x" for Alice's measurements.
    for x_ques in range(num_inputs_alice):
        alice_sum_a = 0
        for a_ans in range(num_outputs_alice):
            alice_sum_a += alice_povms[x_ques, a_ans]
        constraints.append(alice_sum_a == tau)

    constraints.append(cvxpy.trace(tau) == 1)

    problem = cvxpy.Problem(objective, constraints)

    lower_bound = problem.solve()
    return alice_povms, lower_bound


def optimize_bob(dim: int,
                 prob_mat: np.ndarray,
                 pred_mat: np.ndarray,
                 alice_povms) -> Tuple[Dict, float]:
    """Fix Alice's measurements and optimize over Bob's measurements."""
    # Get number of inputs and outputs.
    num_inputs_alice, num_inputs_bob = prob_mat.shape
    num_outputs_alice, num_outputs_bob = pred_mat.shape[0], pred_mat.shape[1]

    # Now, optimize over Bob's measurement operators and fix Alice's operators
    # as those are coming from the previous SDP.
    bob_povms = defaultdict(cvxpy.Variable)
    for y_ques in range(num_inputs_bob):
        for b_ans in range(num_outputs_bob):
            bob_povms[y_ques, b_ans] = cvxpy.Variable((dim, dim), PSD=True)

    win = 0
    for x_ques in range(num_inputs_alice):
        for y_ques in range(num_inputs_bob):
            for a_ans in range(num_outputs_alice):
                for b_ans in range(num_outputs_bob):
                    win += prob_mat[x_ques, y_ques] * \
                           pred_mat[a_ans, b_ans, x_ques, y_ques] * \
                           cvxpy.trace(bob_povms[y_ques, b_ans].H @
                                       alice_povms[x_ques, a_ans].value)

    objective = cvxpy.Maximize(win)
    constraints = []

    # Sum over "b" for all "y" for Bob's measurements.
    for y_ques in range(num_inputs_bob):
        bob_sum_b = 0
        for b_ans in range(num_outputs_bob):
            bob_sum_b += bob_povms[y_ques, b_ans]
        constraints.append(bob_sum_b == np.identity(dim))

    problem = cvxpy.Problem(objective, constraints)

    lower_bound = problem.solve()
    return bob_povms, lower_bound
