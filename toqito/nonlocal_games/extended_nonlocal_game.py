"""Two-player extended nonlocal game."""

from collections import defaultdict

import cvxpy
import numpy as np

from toqito.helper import update_odometer
from toqito.matrix_ops import tensor
from toqito.rand import random_unitary
from toqito.state_opt.npa_hierarchy import npa_constraints


class ExtendedNonlocalGame:
    r"""Create two-player extended nonlocal game object.

    *Extended nonlocal games* are a superset of nonlocal games in which the
    players share a tripartite state with the referee. In such games, the
    winning conditions for Alice and Bob may depend on outcomes of measurements
    made by the referee, on its part of the shared quantum state, in addition
    to Alice and Bob's answers to the questions sent by the referee.

    Extended nonlocal games were initially defined in :footcite:`Johnston_2016_Extended` and more
    information on these games can be found in :footcite:`Russo_2017_Extended`.

    For a detailed walkthrough and several examples, including the BB84 and CHSH
    games, please see the tutorial on :ref:`sphx_glr_auto_examples_extended_nonlocal_games`.

    References
    ==========
    .. footbibliography::


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
                self.dim_x,
                self.dim_y,
                self.num_alice_out,
                self.num_bob_out,
                self.num_alice_in,
                self.num_bob_in,
            ) = pred_mat.shape
            self.prob_mat = tensor(prob_mat, reps)

            pred_mat2 = np.zeros(
                (
                    self.dim_x**reps,
                    self.dim_y**reps,
                    self.num_alice_out**reps,
                    self.num_bob_out**reps,
                    self.num_alice_in**reps,
                    self.num_bob_in**reps,
                )
            )
            i_ind = np.zeros(reps, dtype=int)
            j_ind = np.zeros(reps, dtype=int)
            for i in range(self.num_alice_in**reps):
                for j in range(self.num_bob_in**reps):
                    to_tensor = np.empty([reps, self.dim_x, self.dim_y, self.num_alice_out, self.num_bob_out])
                    for k in range(reps - 1, -1, -1):
                        to_tensor[k] = pred_mat[:, :, :, :, i_ind[k], j_ind[k]]
                    pred_mat2[:, :, :, :, i, j] = tensor(to_tensor)
                    j_ind = update_odometer(j_ind, self.num_bob_in * np.ones(reps))
                i_ind = update_odometer(i_ind, self.num_alice_in * np.ones(reps))
            self.pred_mat = pred_mat2
            self.reps = reps
        self.__get_game_dims()

    def __get_game_dims(self):
        """Initialize game dimensions from the prediction matrix.

        This private method checks whether the game dimensions have already been initialized by
        inspecting the '_dims_initialized_by_get_game_dims' flag. If not, it extracts the dimensions
        from the shape of 'self.pred_mat' and assigns the following instance attributes:

          - referee_dim: The first dimension of self.pred_mat.
          - num_alice_out: The third element of self.pred_mat.shape.
          - num_bob_out: The fourth element.
          - num_alice_in: The fifth element.
          - num_bob_in: The sixth element.

        After extracting these values, the flag '_dims_initialized_by_get_game_dims' is set to True
        to prevent re-initialization on subsequent calls.
        """
        if not hasattr(self, "_dims_initialized_by_get_game_dims") or not self._dims_initialized_by_get_game_dims:
            (
                self.referee_dim,
                _,
                self.num_alice_out,
                self.num_bob_out,
                self.num_alice_in,
                self.num_bob_in,
            ) = self.pred_mat.shape
            self._dims_initialized_by_get_game_dims = True

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

    def quantum_value_lower_bound(
        self,
        iters: int = 20,
        tol: float = 1e-8,
        seed: int = None,
        initial_bob_is_random: bool | dict = False,
        solver: str = cvxpy.SCS,
        solver_params: dict = None,
        verbose: bool = False,
    ) -> float:
        r"""Calculate lower bound on the quantum value of an extended nonlocal game.

        Uses an iterative see-saw method involving two SDPs.

        :param iter: Maximum number of see-saw iterations (Alice optimizes, Bob optimizes (default is 20).
        :param tol: Tolerance for stopping see-saw iteration based on improvement (default is 1e-8).
        :param seed: Optional seed for initializing random POVMs for reproducibility (default is None).
        :param solver: Optional option for different solver (default is SCS).
        :param solver_params: Optional parameters for solver (default is {"eps": 1e-8, "verbose": False}).
        :param verbos: Optional printout for optimizer step (default is False).
        :return: The best lower bound found on the quantum value.
        """
        self.__get_game_dims()
        if solver_params is None:
            solver_params = {"eps_abs": tol, "eps_rel": tol, "max_iters": 50000, "verbose": False}
        if seed is not None:
            np.random.seed(seed)

        # Get number of inputs and outputs for Bob's measurements.
        _, _, _, num_outputs_bob, _, num_inputs_bob = self.pred_mat.shape

        # Initialize Bob's POVMs (NumPy arrays) - Default to RANDOM
        bob_povms_np = defaultdict(lambda: np.zeros((num_outputs_bob, num_outputs_bob), dtype=complex))
        if isinstance(initial_bob_is_random, bool):
            if initial_bob_is_random:
                for y_ques in range(num_inputs_bob):
                    random_u_mat = random_unitary(num_outputs_bob)
                    for b_ans in range(num_outputs_bob):
                        ket = random_u_mat[:, b_ans].reshape(-1, 1)
                        bra = ket.conj().T
                        bob_povms_np[y_ques, b_ans] = ket @ bra
            else:
                for y_ques in range(num_inputs_bob):
                    bob_povms_np[y_ques, 0] = np.eye(num_outputs_bob)
                    for b_other_ans in range(1, num_outputs_bob):
                        bob_povms_np[y_ques, b_other_ans] = np.zeros((num_outputs_bob, num_outputs_bob))
        elif isinstance(initial_bob_is_random, dict):  # If you allow dict for custom POVMs
            bob_povms_np = initial_bob_is_random
        else:
            raise TypeError(
                f"Expected initial_bob_is_random to be bool or dict, "  # Adjust if only bool supported
                f"got {type(initial_bob_is_random).__name__} instead."
            )

        prev_win_val = -float("inf")
        current_best_lower_bound = -float("inf")

        if verbose:
            init_bob_display = "dict" if isinstance(initial_bob_is_random, dict) else initial_bob_is_random
            print(
                f"Starting see-saw: max_steps={iters}, tol={tol}, seed={seed}, solver={solver}, "
                + f"random_init={init_bob_display}"
            )

        for step in range(iters):
            opt_alice_rho_cvxpy_vars, problem_alice = self.__optimize_alice(bob_povms_np, solver, solver_params)

            if (
                opt_alice_rho_cvxpy_vars is None
                or problem_alice.status not in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]
                or problem_alice.value is None
            ):
                if verbose:
                    print(
                        f"Warning: Alice optimization step failed (status: {problem_alice.status}) "
                        f"in see-saw step {step + 1}. Value: {problem_alice.value}"
                    )
                return current_best_lower_bound if current_best_lower_bound > -float("inf") else 0.0

            opt_bob_povm_cvxpy_vars, problem_bob = self.__optimize_bob(opt_alice_rho_cvxpy_vars, solver, solver_params)

            if (
                opt_bob_povm_cvxpy_vars is None
                or problem_bob.status not in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]
                or problem_bob.value is None
            ):
                if verbose:
                    print(
                        f"Warning: Bob optimization step failed (status: {problem_bob.status}) "
                        f"in see-saw step {step + 1}. Value: {problem_bob.value}"
                    )
                return current_best_lower_bound if current_best_lower_bound > -float("inf") else 0.0

            current_win_val = problem_bob.value

            current_best_lower_bound = max(current_best_lower_bound, current_win_val)

            improvement = abs(current_win_val - prev_win_val)

            if verbose:
                print(
                    f"See-saw step {step + 1}/{iters}: Win prob = {current_win_val:.8f}, "
                    + f"Improv = {improvement:.2e}, Best = {current_best_lower_bound:.8f}"
                )

            if (
                improvement < tol and step > 0
            ):  # step > 0 to ensure prev_win_val is not -inf for the first real improvement check
                if verbose:
                    print(f"See-saw converged at step {step + 1} with value {current_best_lower_bound:.8f}")
                break
            prev_win_val = current_win_val

            # If not the last iteration, update Bob's POVMs for the next step
            if step < iters - 1:
                for y_idx in range(self.num_bob_in):
                    for b_idx in range(self.num_bob_out):
                        povm_var = opt_bob_povm_cvxpy_vars.get((y_idx, b_idx))  # Use .get for safety
                        if povm_var is None or povm_var.value is None:
                            if verbose:
                                print(
                                    f"Warning: Bob POVM var ({y_idx},{b_idx}) value is None in step {step + 1} "
                                    f"during POVM update. Exiting see-saw early."
                                )
                            return current_best_lower_bound if current_best_lower_bound > -float("inf") else 0.0
                        bob_povms_np[y_idx, b_idx] = povm_var.value

        else:
            if verbose and iters > 0:
                print(f"See-Saw reached max steps ({iters}) with value {current_best_lower_bound:.8f}")

        return current_best_lower_bound if current_best_lower_bound > -float("inf") else 0.0

    def __optimize_alice(
        self, fixed_bob_povms_np: dict, solver: str = cvxpy.SCS, solver_params: dict = None
    ) -> tuple[dict | None, cvxpy.Problem]:
        """Fix Bob's measurements and optimize over Alice's measurements."""
        # The cvxpy package does not support optimizing over 4-dimensional objects.
        # To overcome this, we use a dictionary to index between the questions and
        # answers, while the cvxpy variables held at this positions are
        # `dim`-by-`dim` cvxpy variables.
        self.__get_game_dims()

        rho_xa_cvxpy_vars = defaultdict(cvxpy.Variable)
        for x_ques in range(self.num_alice_in):
            for a_ans in range(self.num_alice_out):
                rho_xa_cvxpy_vars[x_ques, a_ans] = cvxpy.Variable(
                    (self.referee_dim * self.num_bob_out, self.referee_dim * self.num_bob_out),
                    hermitian=True,
                    name=f"rho_A_{x_ques}{a_ans}",
                )
        tau_A_cvxpy_var = cvxpy.Variable(
            (self.referee_dim * self.num_bob_out, self.referee_dim * self.num_bob_out), hermitian=True, name="tau_A"
        )

        win_objective = cvxpy.Constant(0)
        for x_q in range(self.num_alice_in):
            for y_q in range(self.num_bob_in):
                if self.prob_mat[x_q, y_q] == 0:
                    continue
                for a_ans_alice in range(self.num_alice_out):
                    for b_ans_bob in range(self.num_bob_out):
                        # fixed_bob_povms_np is guaranteed to be np.ndarray here
                        v_xyab = self.pred_mat[:, :, a_ans_alice, b_ans_bob, x_q, y_q]
                        b_yb = fixed_bob_povms_np[y_q, b_ans_bob]
                        op_for_trace = np.kron(v_xyab, b_yb)
                        win_objective += self.prob_mat[x_q, y_q] * cvxpy.trace(
                            op_for_trace.conj().T @ rho_xa_cvxpy_vars[x_q, a_ans_alice]
                        )

        objective = cvxpy.Maximize(cvxpy.real(win_objective))
        constraints = []
        for x_q_constr in range(self.num_alice_in):
            rho_sum_a_constr = 0
            for a_ans_constr in range(self.num_alice_out):
                constraints.append(rho_xa_cvxpy_vars[x_q_constr, a_ans_constr] >> 0)
                rho_sum_a_constr += rho_xa_cvxpy_vars[x_q_constr, a_ans_constr]
            constraints.append(rho_sum_a_constr == tau_A_cvxpy_var)
        constraints.append(cvxpy.trace(tau_A_cvxpy_var) == 1)
        constraints.append(tau_A_cvxpy_var >> 0)

        problem = cvxpy.Problem(objective, constraints)
        problem.solve(solver=solver, **solver_params)  # Use passed solver and params

        return rho_xa_cvxpy_vars, problem

    def __optimize_bob(
        self, alice_rho_cvxpy_vars: dict | None, solver: str = cvxpy.SCS, solver_params: dict = None
    ) -> tuple[dict | None, cvxpy.Problem]:
        """Fix Alice's measurements and optimize over Bob's measurements."""
        # Get number of inputs and outputs.
        self.__get_game_dims()

        # The cvxpy package does not support optimizing over 4-dimensional objects.
        # To overcome this, we use a dictionary to index between the questions and
        # answers, while the cvxpy variables held at this positions are
        # `dim`-by-`dim` cvxpy variables.

        bob_povm_cvxpy_vars = defaultdict(cvxpy.Variable)
        for y_ques in range(self.num_bob_in):
            for b_ans in range(self.num_bob_out):
                bob_povm_cvxpy_vars[y_ques, b_ans] = cvxpy.Variable(
                    (self.num_bob_out, self.num_bob_out), hermitian=True, name=f"B_POVM_{y_ques}{b_ans}"
                )

        win_objective = cvxpy.Constant(0)

        for x_q in range(self.num_alice_in):
            for y_q in range(self.num_bob_in):
                if self.prob_mat[x_q, y_q] == 0:
                    continue
                for a_ans_alice in range(self.num_alice_out):
                    rho_xa_val = alice_rho_cvxpy_vars[x_q, a_ans_alice].value

                    for b_ans_bob in range(self.num_bob_out):
                        v_xyab = self.pred_mat[:, :, a_ans_alice, b_ans_bob, x_q, y_q]
                        win_objective += self.prob_mat[x_q, y_q] * cvxpy.trace(
                            cvxpy.kron(v_xyab, bob_povm_cvxpy_vars[y_q, b_ans_bob]) @ rho_xa_val
                        )

        objective = cvxpy.Maximize(cvxpy.real(win_objective))
        constraints = []
        ident_bob_space = np.identity(self.num_bob_out)
        for y_q_constr in range(self.num_bob_in):
            sum_b_povm = 0
            for b_ans_constr in range(self.num_bob_out):
                constraints.append(bob_povm_cvxpy_vars[y_q_constr, b_ans_constr] >> 0)
                sum_b_povm += bob_povm_cvxpy_vars[y_q_constr, b_ans_constr]
            constraints.append(sum_b_povm == ident_bob_space)

        problem = cvxpy.Problem(objective, constraints)
        problem.solve(solver=solver, **solver_params)  # Use passed solver and params

        return bob_povm_cvxpy_vars, problem

    def commuting_measurement_value_upper_bound(self, k: int | str = 1, no_signaling: bool = True) -> float:
        """Compute an upper bound on the commuting measurement value of an extended nonlocal game.

        This function calculates an upper bound on the commuting measurement value by
        using k-levels of the NPA hierarchy :footcite:`Navascues_2008_AConvergent`. The NPA hierarchy is a uniform
        family of semidefinite programs that converges to the commuting measurement value of
        any extended nonlocal game.

        You can determine the level of the hierarchy by a positive integer or a string
        of a form like '1+ab+aab', which indicates that an intermediate level of the hierarchy
        should be used, where this example uses all products of one measurement, all products of
        one Alice and one Bob measurement, and all products of two Alice and one Bob measurements.

        References
        ==========
        .. footbibliography::


        :param k: The level of the NPA hierarchy to use (default=1).
        :return: The upper bound on the commuting strategy value of an extended nonlocal game.

        """
        dR, _, A_out, B_out, A_in, B_in = self.pred_mat.shape
        K = {}
        for x in range(A_in):
            for y in range(B_in):
                K[(x, y)] = cvxpy.Variable((A_out * dR, B_out * dR), hermitian=True, name=f"K({x},{y})")
        total_win = cvxpy.Constant(0)
        for x in range(A_in):
            for y in range(B_in):
                for a in range(A_out):
                    for b in range(B_out):
                        P_ref = self.pred_mat[:, :, a, b, x, y]
                        blk = K[(x, y)][a * dR : (a + 1) * dR, b * dR : (b + 1) * dR]
                        total_win += self.prob_mat[x, y] * cvxpy.trace(P_ref.conj().T @ blk)
        cons = npa_constraints(K, k, referee_dim=dR, no_signaling=no_signaling)
        prob = cvxpy.Problem(cvxpy.Maximize(cvxpy.real(total_win)), cons)
        cs_val = prob.solve(solver=cvxpy.SCS, eps=1e-8, max_iters=100_000, verbose=False)

        return cs_val
