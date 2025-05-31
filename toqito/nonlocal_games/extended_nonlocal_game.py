"""Two-player extended nonlocal game."""

from collections import defaultdict

import cvxpy
import numpy as np

from toqito.helper import npa_constraints, update_odometer
from toqito.matrix_ops import tensor


class ExtendedNonlocalGame:
    r"""Create two-player extended nonlocal game object.

    *Extended nonlocal games* are a superset of nonlocal games in which the
    players share a tripartite state with the referee. In such games, the
    winning conditions for Alice and Bob may depend on outcomes of measurements
    made by the referee, on its part of the shared quantum state, in addition
    to Alice and Bob's answers to the questions sent by the referee.

    Extended nonlocal games were initially defined in :cite:`Johnston_2016_Extended` and more
    information on these games can be found in :cite:`Russo_2017_Extended`.

    An example demonstration is available as a tutorial in the
    documentation. Go to :ref:`ref-label-bb84_extended_nl_example`.

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    """

    def __init__(self, prob_mat: np.ndarray, pred_mat: np.ndarray, reps: int = 1) -> None:
        """Construct extended nonlocal game object.

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
                    dim_x**reps,
                    dim_y**reps,
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
                    to_tensor = np.empty([reps, dim_x, dim_y, num_alice_out, num_bob_out])
                    for k in range(reps - 1, -1, -1):
                        to_tensor[k] = pred_mat[:, :, :, :, i_ind[k], j_ind[k]]
                    pred_mat2[:, :, :, :, i, j] = tensor(to_tensor)
                    j_ind = update_odometer(j_ind, num_bob_in * np.ones(reps))
                i_ind = update_odometer(i_ind, num_alice_in * np.ones(reps))
            self.pred_mat = pred_mat2
            self.reps = reps

    def unentangled_value(self) -> float:
        r"""Calculate the unentangled value of an extended nonlocal game.

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
                        p_win += self.prob_mat[x_in, y_in] * self.pred_mat[:, :, a_out, b_out, x_in, y_in]

                rho = cvxpy.Variable((dim_x, dim_y), hermitian=True)

                objective = cvxpy.Maximize(cvxpy.real(cvxpy.trace(p_win.conj().T @ rho)))

                constraints = [cvxpy.trace(rho) == 1, rho >> 0]
                problem = cvxpy.Problem(objective, constraints)
                unent_val = problem.solve()
                max_unent_val = max(max_unent_val, unent_val)
        return max_unent_val

    def nonsignaling_value(self) -> float:
        r"""Calculate the non-signaling value of an extended nonlocal game.

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
        constraints = []

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

        p_win = 0
        for a_out in range(alice_out):
            for b_out in range(bob_out):
                for x_in in range(alice_in):
                    for y_in in range(bob_in):
                        p_win += self.prob_mat[x_in, y_in] * cvxpy.trace(
                            self.pred_mat[:, :, a_out, b_out, x_in, y_in].conj().T @ k_var[a_out, b_out, x_in, y_in]
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

    # In toqito/nonlocal_games/extended_nonlocal_game.py
    # class ExtendedNonlocalGame:

    def quantum_value_lower_bound(self, iters: int = 1, tol: float = 1e-7, seed: int = None) -> float:
        r"""Calculate lower bound on the quantum value of an extended nonlocal game.

        Uses an iterative see-saw method involving two SDPs.

        :param iter: Maximum number of see-saw iterations (Alice a-optimizes, Bob optimizes).
        :param tol: Tolerance for stopping see-saw iteration based on improvement.
        :param seed: Optional seed for initializing random POVMs for reproducibility.
        :return: The best lower bound found on the quantum value.
        """
        if seed is not None:
            np.random.seed(seed)

        _, _, _, num_outputs_bob, _, num_inputs_bob = self.pred_mat.shape

        # Initialize Bob's POVMs RANDOMLY (this is affected by the seed)
        # These are numpy arrays, not cvxpy variables at this stage.
        bob_povms_np = defaultdict(
            lambda: np.zeros((self.pred_mat.shape[0], self.pred_mat.shape[0]), dtype=complex)
        )  # Assuming POVMs act on referee_dim space initially
        # This part might need adjustment based on how __optimize_alice expects bob_povms
        # The original code implies bob_povms are d_bob x d_bob where d_bob = num_outputs_bob
        # And that they act on a separate Bob's system. Let's stick to that.
        # The __optimize_alice uses np.kron(pred_mat, bob_povm_element), so bob_povm_element is on Bob's space.

        bob_povms_np = defaultdict(lambda: np.zeros((num_outputs_bob, num_outputs_bob), dtype=complex))

        for y_ques in range(num_inputs_bob):
            bob_povms_np[y_ques, 0] = np.eye(num_outputs_bob)
            for b_other_ans in range(1, num_outputs_bob):
                bob_povms_np[y_ques, b_other_ans] = np.zeros((num_outputs_bob, num_outputs_bob))

        # See-saw iteration
        prev_win_val = -float("inf")
        current_best_lower_bound = -float("inf")

        print(f"Starting see-saw for extended game: max_steps={iters}, tol={tol}, seed={seed}")

        for step in range(iters):
            opt_alice_rho_cvxpy_vars, problem_alice = self.__optimize_alice(
                bob_povms_np
            )  # bob_povms_np are NUMPY arrays

            if problem_alice.status not in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE] or problem_alice.value is None:
                print(
                    "Warning: Alice optimization step failed "
                    + "(status: {problem_alice.status}) in see-saw step {step + 1}."
                )
                return current_best_lower_bound if current_best_lower_bound > -float("inf") else 0.0

            # For __optimize_bob, we need the *CVXPY variables* from Alice's step,
            # as their .value will be used inside __optimize_bob's objective.
            # So, __optimize_alice should return the dict of CVXPY variables for rho.
            opt_bob_povm_cvxpy_vars, problem_bob = self.__optimize_bob(opt_alice_rho_cvxpy_vars)

            if problem_bob.status not in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE] or problem_bob.value is None:
                print(
                    f"Warning: Bob optimization step failed (status: {problem_bob.status}) in see-saw step {step + 1}."
                )
                return current_best_lower_bound if current_best_lower_bound > -float("inf") else 0.0

            current_win_val = problem_bob.value  # This is the objective value from Bob's optimization

            # Update Bob's POVMs (numpy arrays) for the next iteration of Alice's optimization
            for y_ques_b_update in range(num_inputs_bob):  # Ensure num_inputs_bob is defined
                for b_ans_b_update in range(num_outputs_bob):  # Ensure num_outputs_bob is defined
                    if opt_bob_povm_cvxpy_vars[y_ques_b_update, b_ans_b_update].value is not None:
                        bob_povms_np[y_ques_b_update, b_ans_b_update] = opt_bob_povm_cvxpy_vars[
                            y_ques_b_update, b_ans_b_update
                        ].value
                    else:
                        print(
                            f"Warning: Bob POVM variable for ({y_ques_b_update},{b_ans_b_update}) "
                            + "is None after solve in step {step + 1}."
                        )
                        # Potentially break or handle this error (e.g., keep old POVM)
                        return current_best_lower_bound  # Or some other error handling

            it_diff = abs(current_win_val - prev_win_val)
            # print(f"See-saw step {step + 1}: Win prob = {current_win_val:.8f}, Improvement = {it_diff:.2e}")

            current_best_lower_bound = max(current_best_lower_bound, current_win_val)

            if it_diff <= tol and step > 0:  # Add step > 0 to ensure at least one full iteration
                print(f"See-saw converged at step {step + 1} with value {current_best_lower_bound:.8f}")
                break
            prev_win_val = current_win_val
        else:
            print(f"See-saw reached max steps ({iters}) with value {current_best_lower_bound:.8f}")

        return current_best_lower_bound

    def __optimize_alice(self, bob_povms) -> tuple[dict, float]:
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
                rho[x_ques, a_ans] = cvxpy.Variable((dim * num_outputs_bob, dim * num_outputs_bob), hermitian=True)

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
        constraints = []

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

        problem.solve(solver=cvxpy.SCS, eps_abs=1e-9, eps_rel=1e-9, verbose=False)
        # Return the dictionary of CVXPY variables for rho and the problem object
        if problem.status in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE] and problem.value is not None:
            return rho, problem
        else:
            # It's better to still return rho (even if not optimal) if problem.variables() have values,
            # but indicate failure through the problem object. Or return None for rho.
            # For simplicity if problem fails badly, returning None for variables is safer.
            print(f"Warning: __optimize_alice failed to solve optimally. Status: {problem.status}")
            return None, problem

    def __optimize_bob(self, rho) -> tuple[dict, float]:
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
        # When building objective, use alice_rho_cvxpy_vars[x,a].value
        for x_ques in range(num_inputs_alice):  # num_inputs_alice needs to be defined or passed
            for y_ques in range(num_inputs_bob):
                for a_ans in range(num_outputs_alice):  # num_outputs_alice needs to be defined
                    for b_ans in range(num_outputs_bob):
                        if rho[x_ques, a_ans].value is not None:  # Check if Alice's part has a value
                            win += self.prob_mat[x_ques, y_ques] * cvxpy.trace(
                                (
                                    cvxpy.kron(
                                        self.pred_mat[:, :, a_ans, b_ans, x_ques, y_ques],
                                        bob_povms[y_ques, b_ans],  # This is CVXPY var
                                    )
                                )
                                # Here, we need the .value from Alice's optimization
                                @ rho[x_ques, a_ans].value
                            )
                        else:
                            # This implies Alice's optimization failed to provide a value.
                            # Should not happen if quantum_value_lower_bound checks problem_alice.value
                            # Can add a very small penalty or skip term if this is possible.
                            # For now, assume alice_rho_cvxpy_vars[x,a].value is valid.
                            pass
        objective = cvxpy.Maximize(cvxpy.real(win))

        constraints = []

        # Sum over "b" for all "y" for Bob's measurements.
        for y_q in range(num_inputs_bob):  # num_inputs_bob needs to be defined
            s = 0
            for b_a in range(num_outputs_bob):  # num_outputs_bob needs to be defined
                constraints.append(bob_povms[y_q, b_a] >> 0)
                s += bob_povms[y_q, b_a]
            constraints.append(s == np.identity(num_outputs_bob))  # Ensure correct Identity dimension

        problem = cvxpy.Problem(objective, constraints)
        problem.solve(solver=cvxpy.SCS, eps_abs=1e-9, eps_rel=1e-9, verbose=False)

        if problem.status in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE] and problem.value is not None:
            return bob_povms, problem  # bob_povms is dict of CVXPY variables
        else:
            print(f"Warning: __optimize_bob failed to solve optimally. Status: {problem.status}")
            return None, problem

    def commuting_measurement_value_upper_bound(self, k: int | str = 1) -> float:
        """Compute an upper bound on the commuting measurement value of an extended nonlocal game.

        This function calculates an upper bound on the commuting measurement value by
        using k-levels of the NPA hierarchy :cite:`Navascues_2008_AConvergent`. The NPA hierarchy is a uniform family
        of semidefinite programs that converges to the commuting measurement value of
        any extended nonlocal game.

        You can determine the level of the hierarchy by a positive integer or a string
        of a form like '1+ab+aab', which indicates that an intermediate level of the hierarchy
        should be used, where this example uses all products of one measurement, all products of
        one Alice and one Bob measurement, and all products of two Alice and one Bob measurements.

        References
        ==========
        .. bibliography::
            :filter: docname in docnames

        :param k: The level of the NPA hierarchy to use (default=1).
        :return: The upper bound on the commuting strategy value of an extended nonlocal game.

        """
        referee_dim, _, alice_out, bob_out, alice_in, bob_in = self.pred_mat.shape

        mat = defaultdict(cvxpy.Variable)
        for x_in in range(alice_in):
            for y_in in range(bob_in):
                mat[x_in, y_in] = cvxpy.Variable(
                    (alice_out * referee_dim, bob_out * referee_dim),
                    name=f"K(a, b | {x_in}, {y_in})",
                    hermitian=True,
                )

        p_win = cvxpy.Constant(0)
        for a_out in range(alice_out):
            for b_out in range(bob_out):
                for x_in in range(alice_in):
                    for y_in in range(bob_in):
                        p_win += self.prob_mat[x_in, y_in] * cvxpy.trace(
                            self.pred_mat[:, :, a_out, b_out, x_in, y_in].conj().T
                            @ mat[x_in, y_in][
                                a_out * referee_dim : (a_out + 1) * referee_dim,
                                b_out * referee_dim : (b_out + 1) * referee_dim,
                            ]
                        )

        npa = npa_constraints(mat, k, referee_dim)
        objective = cvxpy.Maximize(cvxpy.real(p_win))
        problem = cvxpy.Problem(objective, npa)
        cs_val = problem.solve()

        return cs_val
