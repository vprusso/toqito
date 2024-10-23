"""Two-player nonlocal game."""

from collections import defaultdict

import cvxpy
import numpy as np

from toqito.helper import npa_constraints, update_odometer
from toqito.matrix_ops import tensor
from toqito.rand import random_povm


class NonlocalGame:
    r"""Create two-player nonlocal game object.

    *Nonlocal games* are a mathematical framework that abstractly models a
    physical system. This game is played between two players, Alice and Bob, who
    are not allowed to communicate with each other once the game has started and
    who play cooperative against an adversary referred to as the referee.

    The nonlocal game framework was originally introduced in :cite:`Cleve_2010_Consequences`.

    A tutorial is available in the documentation. For more info, see :ref:`ref-label-nl-games-tutorial`.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    """

    def __init__(self, prob_mat: np.ndarray, pred_mat: np.ndarray, reps: int = 1) -> None:
        """Construct nonlocal game object.

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
                    num_alice_out**reps,
                    num_bob_out**reps,
                    num_alice_in**reps,
                    num_bob_in**reps,
                )
            )
            i_ind = np.zeros(reps, dtype=int)
            j_ind = np.zeros(reps, dtype=int)
            for i in range(num_alice_in**reps):
                for j in range(num_bob_in**reps):
                    to_tensor = np.empty([reps, num_alice_out, num_bob_out])
                    for k in range(reps - 1, -1, -1):
                        to_tensor[k] = pred_mat[:, :, i_ind[k], j_ind[k]]
                    pred_mat2[:, :, i, j] = tensor(to_tensor)
                    j_ind = update_odometer(j_ind, num_bob_in * np.ones(reps))
                i_ind = update_odometer(i_ind, num_alice_in * np.ones(reps))
            self.pred_mat = pred_mat2
            self.reps = reps

    @classmethod
    def from_bcs_game(cls, constraints: list[np.ndarray], reps: int = 1) -> "NonlocalGame":
        """Convert constraints that specify a binary constraint system game to a nonlocal game.

        Binary constraint system games (BCS) games were originally defined in :cite:`Cleve_2014_Characterization`.

        :param constraints: List of binary constraints that define the game.
        :param reps: Number of parallel repetitions to perform. Default is 1.
        :return: A NonlocalGame object arising from the variables and constraints that define the game.
        """
        if (num_constraints := len(constraints)) == 0:
            raise ValueError("At least 1 constraint is required")
        num_variables = constraints[0].ndim

        # Retrieve dependent variables for each constraint.
        dependent_variables = np.zeros((num_constraints, num_variables))

        for j in range(num_constraints):
            for i in range(num_variables):
                # Identifying independent variables based on equality check.
                dependent_variables[j, i] = np.diff(constraints[j], axis=i).any()

        # Compute the probability matrix.
        prob_mat = np.zeros((num_constraints, num_variables))
        for j in range(num_constraints):
            p_x = 1.0 / num_constraints
            num_dependent_vars = dependent_variables[j].sum()
            p_y = dependent_variables[j] / num_dependent_vars
            prob_mat[j] = p_x * p_y

        # Compute the prediction matrix.
        pred_mat = np.zeros((2**num_variables, 2, num_constraints, num_variables))
        for x_ques in range(num_constraints):
            for a_ans in range(pred_mat.shape[0]):
                # Convert Alice's truth assignment to binary.
                bin_a = np.array(list(map(int, np.binary_repr(a_ans, num_variables))))

                # Convert truth assignment to a tuple for easy indexing.
                truth_assignment = tuple(bin_a)

                for y_ques in range(num_variables):
                    # Bob’s assignment is Alice’s truth assignment for the current variable.
                    b_ans = truth_assignment[y_ques]

                    # Check if this satisfies the constraint.
                    if constraints[x_ques][truth_assignment] == 1:
                        pred_mat[a_ans, b_ans, x_ques, y_ques] = 1

        return cls(prob_mat, pred_mat, reps)

    def classical_value(self) -> float:
        """Compute the classical value of the nonlocal game.

        This function has been adapted from the QETLAB package.

        :return: A value between [0, 1] representing the classical value.
        """
        (
            num_alice_outputs,
            num_bob_outputs,
            num_alice_inputs,
            num_bob_inputs,
        ) = self.pred_mat.shape

        # Create a copy of pred_mat to avoid in-place modification
        pred_mat_copy = np.copy(self.pred_mat)

        for x_alice_in in range(num_alice_inputs):
            for y_bob_in in range(num_bob_inputs):
                pred_mat_copy[:, :, x_alice_in, y_bob_in] = (
                    self.prob_mat[x_alice_in, y_bob_in] * pred_mat_copy[:, :, x_alice_in, y_bob_in]
                )
        p_win = float("-inf")
        if num_alice_outputs**num_alice_inputs < num_bob_outputs**num_bob_inputs:
            pred_mat_copy = np.transpose(pred_mat_copy, (1, 0, 3, 2))
            (
                num_alice_outputs,
                num_bob_outputs,
                num_alice_inputs,
                num_bob_inputs,
            ) = pred_mat_copy.shape
        pred_mat_copy = np.transpose(pred_mat_copy, (0, 2, 1, 3))

        for i in range(num_alice_outputs**num_bob_inputs):
            number = i
            base = num_bob_outputs
            digits = num_bob_inputs
            b_ind = np.zeros(digits)
            for j in range(digits):
                b_ind[digits - j - 1] = np.mod(number, base)
                number = np.floor(number / base)
            pred_alice = np.zeros((num_alice_outputs, num_alice_inputs))

            for y_bob_in in range(num_bob_inputs):
                pred_alice = pred_alice + pred_mat_copy[:, :, int(b_ind[y_bob_in]), y_bob_in]
            tgval = np.sum(np.amax(pred_alice, axis=0))
            p_win = max(p_win, tgval)
        return p_win

    def quantum_value_lower_bound(
        self,
        dim: int = 2,
        iters: int = 5,
        tol: float = 10e-6,
    ):
        r"""Compute a lower bound on the quantum value of a nonlocal game :cite:`Liang_2007_Bounds`.

        Calculates a lower bound on the maximum value that the specified
        nonlocal game can take on in quantum mechanical settings where Alice and
        Bob each have access to `dim`-dimensional quantum system.

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
        can be applied to Bell inequalities as shown in :cite:`Liang_2007_Bounds`.

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

        .. math::
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
        >>> pred_mat = np.zeros((num_alice_outputs, num_bob_outputs, num_alice_inputs, num_bob_inputs))
        >>>
        >>> for a_alice in range(num_alice_outputs):
        ...     for b_bob in range(num_bob_outputs):
        ...        for x_alice in range(num_alice_inputs):
        ...            for y_bob in range(num_bob_inputs):
        ...                if np.mod(a_alice + b_bob + x_alice * y_bob, dim) == 0:
        ...                    pred_mat[a_alice, b_bob, x_alice, y_bob] = 1
        >>>
        >>> chsh = NonlocalGame(prob_mat, pred_mat)
        >>> chsh.quantum_value_lower_bound()   # doctest: +SKIP
        0.85

        References
        ==========
        .. bibliography::
            :filter: docname in docnames


        :param dim: The dimension of the quantum system that Alice and Bob have
                    access to (default = 2).
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
                # Optimize over Alice's measurement operators while fixing
                # Bob's. If this is the first iteration, then the previously
                # randomly generated operators in the outer loop are Bob's.
                # Otherwise, Bob's operators come from running the next SDP.
                alice_povms, lower_bound = self.__optimize_alice(dim, bob_povms)
                bob_povms, lower_bound = self.__optimize_bob(dim, alice_povms)

                it_diff = lower_bound - prev_win
                prev_win = lower_bound

                # As the SDPs keep alternating, check if the winning probability
                # becomes any higher. If so, replace with new best.
                best = max(best, lower_bound)

            best_lower_bound = max(best, best_lower_bound)

        return best_lower_bound

    def __optimize_alice(self, dim, bob_povms) -> tuple[dict, float]:
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
                alice_povms[x_ques, a_ans] = cvxpy.Variable((dim, dim), hermitian=True)

        tau = cvxpy.Variable((dim, dim), hermitian=True)

        # .. math::
        #    \sum_{(x,y) \in \Sigma} \pi(x, y) V(a,b|x,y) \ip{B_b^y}{A_a^x}
        win = 0
        for x_ques in range(num_inputs_alice):
            for y_ques in range(num_inputs_bob):
                for a_ans in range(num_outputs_alice):
                    for b_ans in range(num_outputs_bob):
                        if isinstance(bob_povms[y_ques, b_ans], np.ndarray):
                            win += (
                                self.prob_mat[x_ques, y_ques]
                                * self.pred_mat[a_ans, b_ans, x_ques, y_ques]
                                * cvxpy.trace(bob_povms[y_ques, b_ans].conj().T @ alice_povms[x_ques, a_ans])
                            )
                        if isinstance(
                            bob_povms[y_ques, b_ans],
                            cvxpy.expressions.variable.Variable,
                        ):
                            win += (
                                self.prob_mat[x_ques, y_ques]
                                * self.pred_mat[a_ans, b_ans, x_ques, y_ques]
                                * cvxpy.trace(bob_povms[y_ques, b_ans].value.conj().T @ alice_povms[x_ques, a_ans])
                            )

        objective = cvxpy.Maximize(cvxpy.real(win))

        constraints = []

        # Sum over "a" for all "x" for Alice's measurements.
        for x_ques in range(num_inputs_alice):
            alice_sum_a = 0
            for a_ans in range(num_outputs_alice):
                alice_sum_a += alice_povms[x_ques, a_ans]
                constraints.append(alice_povms[x_ques, a_ans] >> 0)
            constraints.append(alice_sum_a == tau)

        constraints.append(cvxpy.trace(tau) == 1)
        constraints.append(tau >> 0)

        problem = cvxpy.Problem(objective, constraints)

        lower_bound = problem.solve()
        return alice_povms, lower_bound

    def __optimize_bob(self, dim, alice_povms) -> tuple[dict, float]:
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
                bob_povms[y_ques, b_ans] = cvxpy.Variable((dim, dim), hermitian=True)

        win = 0
        for x_ques in range(num_inputs_alice):
            for y_ques in range(num_inputs_bob):
                for a_ans in range(num_outputs_alice):
                    for b_ans in range(num_outputs_bob):
                        win += (
                            self.prob_mat[x_ques, y_ques]
                            * self.pred_mat[a_ans, b_ans, x_ques, y_ques]
                            * cvxpy.trace(bob_povms[y_ques, b_ans].H @ alice_povms[x_ques, a_ans].value)
                        )

        objective = cvxpy.Maximize(cvxpy.real(win))
        constraints = []

        # Sum over "b" for all "y" for Bob's measurements.
        for y_ques in range(num_inputs_bob):
            bob_sum_b = 0
            for b_ans in range(num_outputs_bob):
                bob_sum_b += bob_povms[y_ques, b_ans]
                constraints.append(bob_povms[y_ques, b_ans] >> 0)
            constraints.append(bob_sum_b == np.identity(dim))

        problem = cvxpy.Problem(objective, constraints)

        lower_bound = problem.solve()
        return bob_povms, lower_bound

    def nonsignaling_value(self) -> float:
        """Compute the non-signaling value of the nonlocal game.

        :return: A value between [0, 1] representing the non-signaling value.
        """
        alice_out, bob_out, alice_in, bob_in = self.pred_mat.shape
        dim_x, dim_y = 2, 2

        constraints = []

        # Define K(a,b|x,y) variable.
        k_var = defaultdict(cvxpy.Variable)
        for a_out in range(alice_out):
            for b_out in range(bob_out):
                for x_in in range(alice_in):
                    for y_in in range(bob_in):
                        k_var[a_out, b_out, x_in, y_in] = cvxpy.Variable((dim_x, dim_y), hermitian=True)
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

        p_win = cvxpy.Constant(0)
        for a_out in range(alice_out):
            for b_out in range(bob_out):
                for x_in in range(alice_in):
                    for y_in in range(bob_in):
                        p_win += self.prob_mat[x_in, y_in] * cvxpy.trace(
                            self.pred_mat[a_out, b_out, x_in, y_in].conj().T * k_var[a_out, b_out, x_in, y_in]
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

    def commuting_measurement_value_upper_bound(self, k: int | str = 1) -> float:
        """Compute an upper bound on the commuting measurement value of the nonlocal game.

        This function calculates an upper bound on the commuting measurement value by
        using k-levels of the NPA hierarchy :cite:`Navascues_2008_AConvergent`. The NPA hierarchy is a uniform family
        of semidefinite programs that converges to the commuting measurement value of
        any nonlocal game.

        You can determine the level of the hierarchy by a positive integer or a string
        of a form like '1+ab+aab', which indicates that an intermediate level of the hierarchy
        should be used, where this example uses all products of one measurement, all products of
        one Alice and one Bob measurement, and all products of two Alice and one Bob measurements.

        References
        ==========
        .. bibliography::
            :filter: docname in docnames

        :param k: The level of the NPA hierarchy to use (default=1).
        :return: The upper bound on the commuting strategy value of a nonlocal game.

        """
        alice_out, bob_out, alice_in, bob_in = self.pred_mat.shape

        mat = defaultdict(cvxpy.Variable)
        for x_in in range(alice_in):
            for y_in in range(bob_in):
                mat[x_in, y_in] = cvxpy.Variable((alice_out, bob_out), name=f"M(a, b | {x_in}, {y_in})")

        p_win = cvxpy.Constant(0)
        for a_out in range(alice_out):
            for b_out in range(bob_out):
                for x_in in range(alice_in):
                    for y_in in range(bob_in):
                        p_win += (
                            self.prob_mat[x_in, y_in]
                            * self.pred_mat[a_out, b_out, x_in, y_in]
                            * mat[x_in, y_in][a_out, b_out]
                        )

        npa = npa_constraints(mat, k)
        objective = cvxpy.Maximize(p_win)
        problem = cvxpy.Problem(objective, npa)
        cs_val = problem.solve()

        return cs_val
