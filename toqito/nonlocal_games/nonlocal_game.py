"""Two-player nonlocal game."""
from typing import Dict, Tuple
from collections import defaultdict
import cvxpy
import numpy as np
from toqito.random.random_povm import random_povm


class NonlocalGame:
    r"""
    """
    def __init__(self,
                 dim: int,
                 prob_mat: np.ndarray,
                 pred_mat: np.ndarray,
                 iters: int = 5,
                 tol: float = 10e-6) -> None:

        self.dim = dim
        self.prob_mat = prob_mat
        self.pred_mat = pred_mat
        self.iters = iters
        self.tol = tol

        if -np.min(np.min(self.prob_mat)) > self.tol:
            raise ValueError(
                "Invalid: The variable `prob_mat` must be a "
                "probability matrix: its entries must be "
                "non-negative."
            )
        if np.abs(np.sum(np.sum(self.prob_mat)) - 1) > self.tol:
            raise ValueError(
                "Invalid: The variable `prob_mat` must be a "
                "probability matrix: its entries must sum to 1."
            )

    def classical_value(self) -> float:
        """
        Compute the classical value of the nonlocal game.

        :return: A value between [0, 1] representing the classical value.
        """
        num_alice_outputs, num_bob_outputs, \
            num_alice_inputs, num_bob_inputs = self.pred_mat.shape

        p_win = 0
        for a in range(num_alice_outputs):
            for b in range(num_bob_outputs):
                p_sum = 0
                for x in range(num_alice_inputs):
                    for y in range(num_bob_inputs):
                        p_sum += self.prob_mat[x, y] * self.pred_mat[a, b, x, y]
                p_win = max(p_win, p_sum)
        return p_win

    def quantum_value_lower_bound(self):
        """
        :return:
        """
        # Get number of inputs and outputs.
        num_inputs_bob = self.prob_mat.shape[1]
        num_outputs_bob = self.pred_mat.shape[1]

        best_lower_bound = float("-inf")
        for _ in range(self.iters):
            # Generate a set of random POVMs for Bob. These measurements serve as a
            # rough starting point for the alternating projection algorithm.
            bob_tmp = random_povm(self.dim, num_inputs_bob, num_outputs_bob)
            bob_povms = defaultdict(int)
            for y_ques in range(num_inputs_bob):
                for b_ans in range(num_outputs_bob):
                    bob_povms[y_ques, b_ans] = bob_tmp[:, :, y_ques, b_ans]

            # Run the alternating projection algorithm between the two SDPs.
            it_diff = 1
            prev_win = -1
            best = float("-inf")
            while it_diff > self.tol:
                # Optimize over Alice's measurement operators while fixing Bob's.
                # If this is the first iteration, then the previously randomly
                # generated operators in the outer loop are Bob's. Otherwise, Bob's
                # operators come from running the next SDP.
                alice_povms, lower_bound = self.__optimize_alice(bob_povms)
                bob_povms, lower_bound = self.__optimize_bob(alice_povms)

                it_diff = lower_bound - prev_win
                prev_win = lower_bound

                # As the SDPs keep alternating, check if the winning probability
                # becomes any higher. If so, replace with new best.
                if best < lower_bound:
                    best = lower_bound

            if best_lower_bound < best:
                best_lower_bound = best

        return best_lower_bound

    def __optimize_alice(self, bob_povms) -> Tuple[Dict, float]:
        """Fix Bob's measurements and optimize over Alice's measurements."""
        # Get number of inputs and outputs.
        num_inputs_alice, num_inputs_bob = self.prob_mat.shape
        num_outputs_alice, num_outputs_bob = self.pred_mat.shape[0], self.pred_mat.shape[1]

        # The cvxpy package does not support optimizing over 4-dimensional objects.
        # To overcome this, we use a dictionary to index between the questions and
        # answers, while the cvxpy variables held at this positions are
        # `dim`-by-`dim` cvxpy variables.
        alice_povms = defaultdict(cvxpy.Variable)
        for x_ques in range(num_inputs_alice):
            for a_ans in range(num_outputs_bob):
                alice_povms[x_ques, a_ans] = \
                    cvxpy.Variable((self.dim, self.dim), PSD=True)

        tau = cvxpy.Variable((self.dim, self.dim), PSD=True)

        # .. math::
        #    \sum_{(x,y) \in \Sigma} \pi(x, y) V(a,b|x,y) \ip{B_b^y}{A_a^x}
        win = 0
        is_real = True
        for x_ques in range(num_inputs_alice):
            for y_ques in range(num_inputs_bob):
                for a_ans in range(num_outputs_alice):
                    for b_ans in range(num_outputs_bob):
                        if isinstance(bob_povms[y_ques, b_ans], np.ndarray):
                            win += (
                                self.prob_mat[x_ques, y_ques]
                                * self.pred_mat[a_ans, b_ans, x_ques, y_ques]
                                * cvxpy.trace(
                                    bob_povms[y_ques, b_ans].conj().T
                                    @ alice_povms[x_ques, a_ans]
                                )
                            )
                        if isinstance(
                            bob_povms[y_ques, b_ans], cvxpy.expressions.variable.Variable
                        ):
                            is_real = False
                            win += (
                                self.prob_mat[x_ques, y_ques]
                                * self.pred_mat[a_ans, b_ans, x_ques, y_ques]
                                * cvxpy.trace(
                                    bob_povms[y_ques, b_ans].value.conj().T
                                    @ alice_povms[x_ques, a_ans]
                                )
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

    def __optimize_bob(self, alice_povms) -> Tuple[Dict, float]:
        """Fix Alice's measurements and optimize over Bob's measurements."""
        # Get number of inputs and outputs.
        num_inputs_alice, num_inputs_bob = self.prob_mat.shape
        num_outputs_alice, num_outputs_bob = self.pred_mat.shape[0], self.pred_mat.shape[1]

        # Now, optimize over Bob's measurement operators and fix Alice's operators
        # as those are coming from the previous SDP.
        bob_povms = defaultdict(cvxpy.Variable)
        for y_ques in range(num_inputs_bob):
            for b_ans in range(num_outputs_bob):
                bob_povms[y_ques, b_ans] = cvxpy.Variable((self.dim, self.dim), PSD=True)

        win = 0
        for x_ques in range(num_inputs_alice):
            for y_ques in range(num_inputs_bob):
                for a_ans in range(num_outputs_alice):
                    for b_ans in range(num_outputs_bob):
                        win += (
                            self.prob_mat[x_ques, y_ques]
                            * self.pred_mat[a_ans, b_ans, x_ques, y_ques]
                            * cvxpy.trace(
                                bob_povms[y_ques, b_ans].H
                                @ alice_povms[x_ques, a_ans].value
                            )
                        )

        objective = cvxpy.Maximize(win)
        constraints = []

        # Sum over "b" for all "y" for Bob's measurements.
        for y_ques in range(num_inputs_bob):
            bob_sum_b = 0
            for b_ans in range(num_outputs_bob):
                bob_sum_b += bob_povms[y_ques, b_ans]
            constraints.append(bob_sum_b == np.identity(self.dim))

        problem = cvxpy.Problem(objective, constraints)

        lower_bound = problem.solve()
        return bob_povms, lower_bound
