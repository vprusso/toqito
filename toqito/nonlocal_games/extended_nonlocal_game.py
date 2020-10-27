"""Two-player extended nonlocal game."""
from collections import defaultdict
from typing import Dict, Tuple

import cvxpy
import numpy as np

from toqito.matrix_ops import tensor
from toqito.random import random_unitary
from toqito.helper import update_odometer


class ExtendedNonlocalGame:
    r"""
    Create two-player extended nonlocal game object.

    *Extended nonlocal games* are a superset of nonlocal games in which the
    players share a tripartite state with the referee. In such games, the
    winning conditions for Alice and Bob may depend on outcomes of measurements
    made by the referee, on its part of the shared quantum state, in addition
    to Alice and Bob's answers to the questions sent by the referee.

    Extended nonlocal games were initially defined in [JMRW16]_ and more
    information on these games can be found in [Russo17]_.

    References
    ==========
    .. [JMRW16] Nathaniel Johnston, Rajat Mittal, Vincent Russo, John Watrous
        "Extended non-local games and monogamy-of-entanglement games",
        Proceedings of the Royal Society A: Mathematical, Physical and
        Engineering Sciences 472.2189 (2016),
        https://arxiv.org/abs/1510.02083

    .. [Russo17] Vincent Russo
        "Extended nonlocal games"
        https://arxiv.org/abs/1704.07375
    """

    def __init__(self, prob_mat: np.ndarray, pred_mat: np.ndarray, reps: int = 1) -> None:
        """
        Construct extended nonlocal game object.

        :param prob_mat: A matrix whose (x, y)-entry gives the probability
                        that the referee will give Alice the value `x` and Bob
                        the value `y`.
        :param pred_mat:
        :param reps: Number of parallel repetitions to perform.
        """
        if reps == 1:
            self.prob_mat = prob_mat
            self.pred_mat = pred_mat
            self.reps = reps

        else:
            (
                dim_x,
                dim_y,
                num_alice_out,
                num_bob_out,
                num_alice_in,
                num_bob_in,
            ) = pred_mat.shape
            self.prob_mat = tensor(prob_mat, reps)

            pred_mat2 = np.zeros(
                (
                    dim_x ** reps,
                    dim_y ** reps,
                    num_alice_out ** reps,
                    num_bob_out ** reps,
                    num_alice_in ** reps,
                    num_bob_in ** reps,
                )
            )
            i_ind = np.zeros(reps, dtype=int)
            j_ind = np.zeros(reps, dtype=int)
            for i in range(num_alice_in ** reps):
                for j in range(num_bob_in ** reps):
                    to_tensor = np.empty([reps, dim_x, dim_y, num_alice_out, num_bob_out])
                    for k in range(reps - 1, -1, -1):
                        to_tensor[k] = pred_mat[:, :, :, :, i_ind[k], j_ind[k]]
                    pred_mat2[:, :, :, :, i, j] = tensor(to_tensor)
                    j_ind = update_odometer(j_ind, num_bob_in * np.ones(reps))
                i_ind = update_odometer(i_ind, num_alice_in * np.ones(reps))
            self.pred_mat = pred_mat2
            self.reps = reps

    def unentangled_value(self) -> float:
        r"""
        Calculate the unentangled value of an extended nonlocal game.

        The *unentangled value* of an extended nonlocal game is the supremum
        value for Alice and Bob's winning probability in the game over all
        unentangled strategies. Due to convexity and compactness, it is possible
        to calculate the unentangled extended nonlocal game by:

        .. math::
            \omega(G) = \max_{f, g}
            \lVert
            \sum_{(x,y) \in \Sigma_A \times \Sigma_B} \pi(x,y)
            V(f(x), g(y)|x, y)
            \rVert

        where the maximum is over all functions :math:`f : \Sigma_A \rightarrow
        \Gamma_A` and :math:`g : \Sigma_B \rightarrow \Gamma_B`.

        :return: The unentangled value of the extended nonlocal game.
        """
        dim_x, dim_y, alice_out, bob_out, alice_in, bob_in = self.pred_mat.shape

        max_unent_val = float("-inf")
        for a_out in range(alice_out):
            for b_out in range(bob_out):
                p_win = np.zeros([dim_x, dim_y], dtype=complex)
                for x_in in range(alice_in):
                    for y_in in range(bob_in):
                        p_win += (
                            self.prob_mat[x_in, y_in]
                            * self.pred_mat[:, :, a_out, b_out, x_in, y_in]
                        )

                rho = cvxpy.Variable((dim_x, dim_y), hermitian=True)

                objective = cvxpy.Maximize(cvxpy.real(cvxpy.trace(p_win.conj().T @ rho)))

                constraints = [cvxpy.trace(rho) == 1, rho >> 0]
                problem = cvxpy.Problem(objective, constraints)
                unent_val = problem.solve()
                max_unent_val = max(max_unent_val, unent_val)
        return max_unent_val

    def nonsignaling_value(self) -> float:
        r"""
        Calculate the non-signaling value of an extended nonlocal game.

        The *non-signaling value* of an extended nonlocal game is the supremum
        value of the winning probability of the game taken over all
        non-signaling strategies for Alice and Bob.

        A *non-signaling strategy* for an extended nonlocal game consists of a
        function

        .. math::
            K : \Gamma_A \times \Gamma_B \times \Sigma_A \times \Sigma_B
            \rightarrow \text{Pos}(\mathcal{R})

        such that

        .. math::
            \sum_{a \in \Gamma_A} K(a,b|x,y) = \rho_b^y
            \quad \text{and} \quad
            \sum_{b \in \Gamma_B} K(a,b|x,y) = \sigma_a^x,

        for all :math:`x \in \Sigma_A` and :math:`y \in \Sigma_B` where
        :math:`\{\rho_b^y : y \in \Sigma_A, \ b \in \Gamma_B\}` and
        :math:`\{\sigma_a^x : x \in \Sigma_A, \ a \in \Gamma_B\}` are
        collections of operators satisfying

        .. math::
            \sum_{a \in \Gamma_A} \rho_b^y =
            \tau =
            \sum_{b \in \Gamma_B} \sigma_a^x,

        for every choice of :math:`x \in \Sigma_A` and :math:`y \in \Sigma_B`
        where :math:`\tau \in \text{D}(\mathcal{R})` is a density operator.

        :return: The non-signaling value of the extended nonlocal game.
        """
        dim_x, dim_y, alice_out, bob_out, alice_in, bob_in = self.pred_mat.shape
        constraints = list()

        # The cvxpy package does not support optimizing over more than
        # 2-dimensional objects. To overcome this, we use a dictionary to index
        # between the questions and answers, while the cvxpy variables held at
        # this positions are `dim_x`-by-`dim_y` cvxpy Variable objects.

        # Define K(a,b|x,y) variable.
        k_var = defaultdict(cvxpy.Variable)
        for a_out in range(alice_out):
            for b_out in range(bob_out):
                for x_in in range(alice_in):
                    for y_in in range(bob_in):
                        k_var[a_out, b_out, x_in, y_in] = cvxpy.Variable(
                            (dim_x, dim_y), hermitian=True
                        )
                        constraints.append(k_var[a_out, b_out, x_in, y_in] >> 0)

        # Define \sigma_a^x variable.
        sigma = defaultdict(cvxpy.Variable)
        for a_out in range(alice_out):
            for x_in in range(alice_in):
                sigma[a_out, x_in] = cvxpy.Variable((dim_x, dim_y), hermitian=True)

        # Define \rho_b^y variable.
        rho = defaultdict(cvxpy.Variable)
        for b_out in range(bob_out):
            for y_in in range(bob_in):
                rho[b_out, y_in] = cvxpy.Variable((dim_x, dim_y), hermitian=True)

        # Define \tau density operator variable.
        tau = cvxpy.Variable((dim_x, dim_y), hermitian=True)

        p_win = 0
        for a_out in range(alice_out):
            for b_out in range(bob_out):
                for x_in in range(alice_in):
                    for y_in in range(bob_in):
                        p_win += self.prob_mat[x_in, y_in] * cvxpy.trace(
                            self.pred_mat[:, :, a_out, b_out, x_in, y_in].conj().T
                            @ k_var[a_out, b_out, x_in, y_in]
                        )

        objective = cvxpy.Maximize(cvxpy.real(p_win))

        # The following constraints enforce the so-called non-signaling
        # constraints.

        # Enforce that:
        # \sum_{b \in \Gamma_B} K(a,b|x,y) = \sigma_a^x
        for x_in in range(alice_in):
            for y_in in range(bob_in):
                for a_out in range(alice_out):
                    b_sum = 0
                    for b_out in range(bob_out):
                        b_sum += k_var[a_out, b_out, x_in, y_in]
                    constraints.append(b_sum == sigma[a_out, x_in])

        # Enforce non-signaling constraints on Alice marginal:
        # \sum_{a \in \Gamma_A} K(a,b|x,y) = \rho_b^y
        for x_in in range(alice_in):
            for y_in in range(bob_in):
                for b_out in range(bob_out):
                    a_sum = 0
                    for a_out in range(alice_out):
                        a_sum += k_var[a_out, b_out, x_in, y_in]
                    constraints.append(a_sum == rho[b_out, y_in])

        # Enforce non-signaling constraints on Bob marginal:
        # \sum_{a \in \Gamma_A} \sigma_a^x = \tau
        for x_in in range(alice_in):
            sig_a_sum = 0
            for a_out in range(alice_out):
                sig_a_sum += sigma[a_out, x_in]
            constraints.append(sig_a_sum == tau)

        # Enforce that:
        # \sum_{b \in \Gamma_B} \rho_b^y = \tau
        for y_in in range(bob_in):
            rho_b_sum = 0
            for b_out in range(bob_out):
                rho_b_sum += rho[b_out, y_in]
            constraints.append(rho_b_sum == tau)

        # Enforce that tau is a density operator.
        constraints.append(cvxpy.trace(tau) == 1)
        constraints.append(tau >> 0)

        problem = cvxpy.Problem(objective, constraints)
        ns_val = problem.solve()

        return ns_val

    def quantum_value_lower_bound(self, iters: int = 5, tol: float = 10e-6) -> float:
        r"""
        Calculate lower bound on the quantum value of an extended nonlocal game.

        Test

        :return: The quantum value of the extended nonlocal game.
        """
        # Get number of inputs and outputs for Bob's measurements.
        _, _, _, num_outputs_bob, _, num_inputs_bob = self.pred_mat.shape

        best_lower_bound = float("-inf")
        for _ in range(iters):
            # Generate a set of random POVMs for Bob. These measurements serve as a
            # rough starting point for the alternating projection algorithm.
            bob_povms = defaultdict(int)
            for y_ques in range(num_inputs_bob):
                random_mat = random_unitary(num_outputs_bob)
                for b_ans in range(num_outputs_bob):
                    random_mat_trans = random_mat[:, b_ans].conj().T.reshape(-1, 1)
                    bob_povms[y_ques, b_ans] = random_mat[:, b_ans] * random_mat_trans

            # Run the alternating projection algorithm between the two SDPs.
            it_diff = 1
            prev_win = -1
            best = float("-inf")
            while it_diff > tol:
                # Optimize over Alice's measurement operators while fixing Bob's.
                # If this is the first iteration, then the previously randomly
                # generated operators in the outer loop are Bob's. Otherwise, Bob's
                # operators come from running the next SDP.
                rho, lower_bound = self.__optimize_alice(bob_povms)
                bob_povms, lower_bound = self.__optimize_bob(rho)

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
        (
            dim,
            _,
            num_outputs_alice,
            num_outputs_bob,
            num_inputs_alice,
            num_inputs_bob,
        ) = self.pred_mat.shape

        # The cvxpy package does not support optimizing over 4-dimensional objects.
        # To overcome this, we use a dictionary to index between the questions and
        # answers, while the cvxpy variables held at this positions are
        # `dim`-by-`dim` cvxpy variables.
        rho = defaultdict(cvxpy.Variable)
        for x_ques in range(num_inputs_alice):
            for a_ans in range(num_outputs_alice):
                rho[x_ques, a_ans] = cvxpy.Variable(
                    (dim * num_outputs_bob, dim * num_outputs_bob), hermitian=True
                )

        tau = cvxpy.Variable((dim * num_outputs_bob, dim * num_outputs_bob), hermitian=True)
        win = 0
        for x_ques in range(num_inputs_alice):
            for y_ques in range(num_inputs_bob):
                for a_ans in range(num_outputs_alice):
                    for b_ans in range(num_outputs_bob):
                        if isinstance(bob_povms[y_ques, b_ans], np.ndarray):
                            win += self.prob_mat[x_ques, y_ques] * cvxpy.trace(
                                (
                                    np.kron(
                                        self.pred_mat[:, :, a_ans, b_ans, x_ques, y_ques],
                                        bob_povms[y_ques, b_ans],
                                    )
                                )
                                .conj()
                                .T
                                @ rho[x_ques, a_ans]
                            )
                        if isinstance(
                            bob_povms[y_ques, b_ans],
                            cvxpy.expressions.variable.Variable,
                        ):
                            win += self.prob_mat[x_ques, y_ques] * cvxpy.trace(
                                (
                                    np.kron(
                                        self.pred_mat[:, :, a_ans, b_ans, x_ques, y_ques],
                                        bob_povms[y_ques, b_ans].value,
                                    )
                                )
                                .conj()
                                .T
                                @ rho[x_ques, a_ans]
                            )
        objective = cvxpy.Maximize(cvxpy.real(win))
        constraints = list()

        # Sum over "a" for all "x" for Alice's measurements.
        for x_ques in range(num_inputs_alice):
            rho_sum_a = 0
            for a_ans in range(num_outputs_alice):
                rho_sum_a += rho[x_ques, a_ans]
                constraints.append(rho[x_ques, a_ans] >> 0)
            constraints.append(rho_sum_a == tau)

        constraints.append(cvxpy.trace(tau) == 1)
        constraints.append(tau >> 0)

        problem = cvxpy.Problem(objective, constraints)

        lower_bound = problem.solve()
        return rho, lower_bound

    def __optimize_bob(self, rho) -> Tuple[Dict, float]:
        """Fix Alice's measurements and optimize over Bob's measurements."""
        # Get number of inputs and outputs.
        (
            dim,
            _,
            num_outputs_alice,
            num_outputs_bob,
            num_inputs_alice,
            num_inputs_bob,
        ) = self.pred_mat.shape

        # The cvxpy package does not support optimizing over 4-dimensional objects.
        # To overcome this, we use a dictionary to index between the questions and
        # answers, while the cvxpy variables held at this positions are
        # `dim`-by-`dim` cvxpy variables.
        bob_povms = defaultdict(cvxpy.Variable)
        for y_ques in range(num_inputs_bob):
            for b_ans in range(num_outputs_bob):
                bob_povms[y_ques, b_ans] = cvxpy.Variable((dim, dim), hermitian=True)
        win = 0
        for x_ques in range(num_inputs_alice):
            for y_ques in range(num_inputs_bob):
                for a_ans in range(num_outputs_alice):
                    for b_ans in range(num_outputs_bob):
                        win += self.prob_mat[x_ques, y_ques] * cvxpy.trace(
                            (
                                cvxpy.kron(
                                    self.pred_mat[:, :, a_ans, b_ans, x_ques, y_ques],
                                    bob_povms[y_ques, b_ans],
                                )
                            )
                            @ rho[x_ques, a_ans].value
                        )
        objective = cvxpy.Maximize(cvxpy.real(win))

        constraints = list()

        # Sum over "b" for all "y" for Bob's measurements.
        for y_ques in range(num_inputs_bob):
            bob_sum_b = 0
            for b_ans in range(num_outputs_bob):
                bob_sum_b += bob_povms[y_ques, b_ans]
                constraints.append(bob_povms[y_ques, b_ans] >> 0)
            constraints.append(bob_sum_b == np.identity(num_outputs_bob))

        problem = cvxpy.Problem(objective, constraints)

        lower_bound = problem.solve()
        return bob_povms, lower_bound
