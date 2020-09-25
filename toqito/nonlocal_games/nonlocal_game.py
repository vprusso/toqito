"""Two-player nonlocal game."""
from typing import Dict, Tuple
from collections import defaultdict
import cvxpy
import numpy as np
from toqito.random import random_povm
from toqito.matrix_ops import tensor
from toqito.helper import update_odometer


class NonlocalGame:
    r"""
    Create two-player nonlocal game object.

    *Nonlocal games* are a mathematical framework that abstractly models a
    physical system. This game is played between two players, Alice and Bob, who
    are not allowed to communicate with each other once the game has started and
    who play cooperative against an adversary referred to as the referee.

    The nonlocal game framework was originally introduced in [CHTW04_2]_.

    References
    ==========
    .. [CHTW04_2] Cleve, Richard, Hoyer, Peter, Toner, Benjamin, and Watrous, John
        "Consequences and limits of nonlocal strategies"
        Computational Complexity 2004. Proceedings. 19th IEEE Annual Conference.
        https://arxiv.org/abs/quant-ph/0404076
    """

    def __init__(
        self, prob_mat: np.ndarray, pred_mat: np.ndarray, reps: int = 1
    ) -> None:
        """
        Construct nonlocal game object.

        :param prob_mat: A matrix whose (x, y)-entry gives the probability
                        that the referee will give Alice the value `x` and Bob
                        the value `y`.
        :param pred_mat: A four-dimensional matrix whose (a,b,x,y)-entry gives
                         the outcome for answers "a" and "b" given questions
                         "x" and "y".
        :param reps: Number of parallel repetitions to perform. Default is 1.
        """
        if reps == 1:
            self.prob_mat = prob_mat
            self.pred_mat = pred_mat
            self.reps = reps

        else:
            num_alice_out, num_bob_out, num_alice_in, num_bob_in = pred_mat.shape
            self.prob_mat = tensor(prob_mat, reps)

            pred_mat2 = np.zeros(
                (
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
                    to_tensor = np.empty([reps, num_alice_out, num_bob_out])
                    for k in range(reps - 1, -1, -1):
                        to_tensor[k] = pred_mat[:, :, i_ind[k], j_ind[k]]
                    pred_mat2[:, :, i, j] = tensor(to_tensor)
                    j_ind = update_odometer(j_ind, num_bob_in * np.ones(reps))
                i_ind = update_odometer(i_ind, num_alice_in * np.ones(reps))
            self.pred_mat = pred_mat2
            self.reps = reps

    def classical_value(self) -> float:
        """
        Compute the classical value of the nonlocal game.

        :return: A value between [0, 1] representing the classical value.
        """
        if self.reps >= 1:
            (
                num_alice_outputs,
                num_bob_outputs,
                num_alice_inputs,
                num_bob_inputs,
            ) = self.pred_mat.shape

            # Calculating the classical value of a nonlocal game, in general, is
            # NP-hard. Our approach here is to simply loop over all possible
            # combinations of pairs of questions and answers and keep track of
            # which combination yields the highest classical value.
            p_win = 0
            for a_alice_out in range(num_alice_outputs):
                for b_bob_out in range(num_bob_outputs):
                    p_sum = 0
                    for x_alice_in in range(num_alice_inputs):
                        for y_bob_in in range(num_bob_inputs):
                            p_sum += (
                                self.prob_mat[x_alice_in, y_bob_in]
                                * self.pred_mat[
                                    a_alice_out, b_bob_out, x_alice_in, y_bob_in
                                ]
                            )
                    p_win = max(p_win, p_sum)
            return p_win
        raise ValueError(
            "Error: toqito currently does not support multiple "
            "repetitions for the classical value of a nonlocal game."
        )

    def quantum_value_lower_bound(
        self, iters: int = 5, tol: float = 10e-6,
    ):
        r"""
        Compute a lower bound on the quantum value of a nonlocal game [LD07]_.

        Calculates a lower bound on the maximum value that the specified
        nonlocal game can take on in quantum mechanical settings where Alice and
        Bob each have access to `d`-dimensional quantum system.

        This function works by starting with a randomly-generated POVM for Bob,
        and then optimizing Alice's POVM and the shared entangled state. Then
        Alice's POVM and the entangled state are fixed, and Bob's POVM is
        optimized. And so on, back and forth between Alice and Bob until
        convergence is reached.

        Note that the algorithm is not guaranteed to obtain the optimal local
        bound and can get stuck in local minimum values. The alleviate this, the
        `iter` parameter allows one to run the algorithm some pre-specified
        number of times and keep the highest value obtained.

        The algorithm is based on the alternating projections algorithm as it
        can be applied to Bell inequalities as shown in [LD07]_.

        The alternating projection algorithm has also been referred to as the
        "see-saw" algorithm as it goes back and forth between the following two
        semidefinite programs:

        .. math::

            \begin{equation}
                \begin{aligned}
                    \textbf{SDP-1:} \quad & \\
                    \text{maximize:} \quad & \sum_{(x,y \in \Sigma)} \pi(x,y)
                                             \sum_{(a,b) \in \Gamma}
                                             V(a,b|x,y)
                                             \langle B_b^y, A_a^x \rangle \\
                    \text{subject to:} \quad & \sum_{a \in \Gamma_{\mathsf{A}}}=
                                        \tau, \qquad \qquad
                                        \forall x \in \Sigma_{\mathsf{A}}, \\
                                       \quad & A_a^x \in \text{Pos}(\mathcal{A}),
                                        \qquad
                                        \forall x \in \Sigma_{\mathsf{A}}, \
                                        \forall a \in \Gamma_{\mathsf{A}}, \\
                                        & \tau \in \text{D}(\mathcal{A}).
                \end{aligned}
            \end{equation}

        .. math::

            \begin{equation}
                \begin{aligned}
                    \textbf{SDP-2:} \quad & \\
                    \text{maximize:} \quad & \sum_{(x,y \in \Sigma)} \pi(x,y)
                                             \sum_{(a,b) \in \Gamma} V(a,b|x,y)
                                             \langle B_b^y, A_a^x \rangle \\
                    \text{subject to:} \quad & \sum_{b \in \Gamma_{\mathsf{B}}}=
                                        \mathbb{I}, \qquad \qquad
                                        \forall y \in \Sigma_{\mathsf{B}}, \\
                                    \quad & B_b^y \in \text{Pos}(\mathcal{B}),
                                    \qquad \forall y \in \Sigma_{\mathsf{B}}, \
                                    \forall b \in \Gamma_{\mathsf{B}}.
                \end{aligned}
            \end{equation}

        Examples
        ==========

        The CHSH game

        The CHSH game is a two-player nonlocal game with the following
        probability distribution and question and answer sets.

        .. math::
            \begin{equation}
            \begin{aligned}
              \pi(x,y) = \frac{1}{4}, \qquad (x,y) \in \Sigma_A \times \Sigma_B,
              \qquad \text{and} \qquad (a, b) \in \Gamma_A \times \Gamma_B,
            \end{aligned}
            \end{equation}

        where

        .. math::
            \begin{equation}
            \Sigma_A = \{0, 1\}, \quad \Sigma_B = \{0, 1\}, \quad \Gamma_A =
            \{0,1\}, \quad \text{and} \quad \Gamma_B = \{0, 1\}.
            \end{equation}

        Alice and Bob win the CHSH game if and only if the following equation is
        satisfied.

        ..math::
            \begin{equation}
            a \oplus b = x \land y.
            \end{equation}

        Recall that :math:`\oplus` refers to the XOR operation.

        The optimal quantum value of CHSH is
        :math:`\cos(\pi/8)^2 \approx 0.8536` where the optimal classical value
        is :math:`3/4`.

        >>> import numpy as np
        >>> from toqito.nonlocal_games.nonlocal_game import NonlocalGame
        >>>
        >>> dim = 2
        >>> num_alice_inputs, num_alice_outputs = 2, 2
        >>> num_bob_inputs, num_bob_outputs = 2, 2
        >>> prob_mat = np.array([[1 / 4, 1 / 4], [1 / 4, 1 / 4]])
        >>> pred_mat = np.zeros(
        >>>     (num_alice_outputs, num_bob_outputs, num_alice_inputs, num_bob_inputs)
        >>> )
        >>>
        >>> for a_alice in range(num_alice_outputs):
        >>>    for b_bob in range(num_bob_outputs):
        >>>        for x_alice in range(num_alice_inputs):
        >>>            for y_bob in range(num_bob_inputs):
        >>>                if np.mod(a_alice + b_bob + x_alice * y_bob, dim) == 0:
        >>>                    pred_mat[a_alice, b_bob, x_alice, y_bob] = 1
        >>>
        >>> chsh = NonlocalGame(prob_mat, pred_mat)
        >>> chsh.quantum_value_lower_bound()
        0.8535533840915605

        References
        ==========
        .. [LD07] Liang, Yeong-Cherng, and Andrew C. Doherty.
            "Bounds on quantum correlations in Bell-inequality experiments."
            Physical Review A 75.4 (2007): 042103.
            https://arxiv.org/abs/quant-ph/0608128

        :param iters: The number of times to run the alternating projection
                      algorithm.
        :param tol: The tolerance before quitting out of the alternating
                    projection semidefinite program.
        :return: The lower bound on the quantum value of a nonlocal game.
        """
        # Get number of inputs and outputs.
        _, num_outputs_bob, _, num_inputs_bob = self.pred_mat.shape

        best_lower_bound = float("-inf")
        for _ in range(iters):
            # Generate a set of random POVMs for Bob. These measurements serve
            # as a rough starting point for the alternating projection
            # algorithm.
            bob_tmp = random_povm(num_outputs_bob, num_inputs_bob, num_outputs_bob)
            bob_povms = defaultdict(int)
            for y_ques in range(num_inputs_bob):
                for b_ans in range(num_outputs_bob):
                    bob_povms[y_ques, b_ans] = bob_tmp[:, :, y_ques, b_ans]

            # Run the alternating projection algorithm between the two SDPs.
            it_diff = 1
            prev_win = -1
            best = float("-inf")
            while it_diff > tol:
                # Optimize over Alice's measurement operators while fixing
                # Bob's. If this is the first iteration, then the previously
                # randomly generated operators in the outer loop are Bob's.
                # Otherwise, Bob's operators come from running the next SDP.
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
        (
            num_outputs_alice,
            num_outputs_bob,
            num_inputs_alice,
            num_inputs_bob,
        ) = self.pred_mat.shape

        # The cvxpy package does not support optimizing over 4-dimensional
        # objects. To overcome this, we use a dictionary to index between the
        # questions and answers, while the cvxpy variables held at this
        # positions are `dim`-by-`dim` cvxpy variables.
        alice_povms = defaultdict(cvxpy.Variable)
        for x_ques in range(num_inputs_alice):
            for a_ans in range(num_outputs_alice):
                alice_povms[x_ques, a_ans] = cvxpy.Variable(
                    (num_outputs_alice, num_inputs_alice), PSD=True
                )

        tau = cvxpy.Variable((num_outputs_alice, num_outputs_bob), PSD=True)

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
                            bob_povms[y_ques, b_ans],
                            cvxpy.expressions.variable.Variable,
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
        (
            num_outputs_alice,
            num_outputs_bob,
            num_inputs_alice,
            num_inputs_bob,
        ) = self.pred_mat.shape

        # Now, optimize over Bob's measurement operators and fix Alice's
        # operators as those are coming from the previous SDP.
        bob_povms = defaultdict(cvxpy.Variable)
        for y_ques in range(num_inputs_bob):
            for b_ans in range(num_outputs_bob):
                bob_povms[y_ques, b_ans] = cvxpy.Variable(
                    (num_outputs_bob, num_outputs_bob), PSD=True
                )

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
            constraints.append(bob_sum_b == np.identity(num_outputs_bob))

        problem = cvxpy.Problem(objective, constraints)

        lower_bound = problem.solve()
        return bob_povms, lower_bound

    def nonsignaling_value(self) -> float:
        """
        Compute the non-signaling value of the nonlocal game.

        :return: A value between [0, 1] representing the non-signaling value.
        """
        alice_out, bob_out, alice_in, bob_in = self.pred_mat.shape

        # Define C(a,b|x,y) variable.
        c_var = defaultdict(cvxpy.Variable)
        for a_out in range(alice_out):
            for b_out in range(bob_out):
                for x_in in range(alice_in):
                    for y_in in range(bob_in):
                        c_var[a_out, b_out, x_in, y_in] = cvxpy.Variable(
                            (alice_out, bob_out), PSD=True
                        )

        a_sum_var = cvxpy.Variable()
        b_sum_var = cvxpy.Variable()

        p_win = cvxpy.Constant(0)
        for a_out in range(alice_out):
            for b_out in range(bob_out):
                for x_in in range(alice_in):
                    for y_in in range(bob_in):
                        p_win += self.prob_mat[x_in, y_in] * cvxpy.trace(
                            self.pred_mat[a_out, b_out, x_in, y_in].conj().T
                            * c_var[a_out, b_out, x_in, y_in]
                        )

        objective = cvxpy.Maximize(p_win)

        constraints = list()

        # Enforce that probabilities must sum to 1.
        # \sum_{(a,b) \in \Gamma} C(a,b|x,y) = 1
        for x_in in range(alice_in):
            for y_in in range(bob_in):
                a_b_sum = 0
                for a_out in range(alice_out):
                    for b_out in range(bob_out):
                        a_b_sum += c_var[a_out, b_out, x_in, y_in]
                constraints.append(a_b_sum == 1)

        # Enforce non-signaling property on marginal for Alice
        # # \sum_{a \in \Gamma_A} C(a,b|x,y) = \sum_{a \in \Gamma_A} C(a,b|x',y)
        for x_in in range(alice_in):
            for y_in in range(bob_in):
                for b_out in range(bob_out):
                    a_sum = 0
                    for a_out in range(alice_out):
                        a_sum += c_var[a_out, b_out, x_in, y_in]
                    constraints.append(a_sum == a_sum_var)

        # Enforce non-signaling property on marginal for Bob.
        # # \sum_{b \in \Gamma_B} C(a,b|x,y) = \sum_{b \in \Gamma_B} C(a,b|x,y')
        for x_in in range(alice_in):
            for y_in in range(bob_in):
                for a_out in range(alice_out):
                    b_sum = 0
                    for b_out in range(bob_out):
                        b_sum += c_var[a_out, b_out, x_in, y_in]
                    constraints.append(b_sum == b_sum_var)

        problem = cvxpy.Problem(objective, constraints)
        ns_val = 1 / 2 * problem.solve()

        return ns_val
