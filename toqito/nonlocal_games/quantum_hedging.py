"""Semidefinite programs for obtaining values of quantum hedging scenarios."""

import cvxpy
import numpy as np

from toqito._sdp import solve_sdp
from toqito.matrix_ops import partial_trace
from toqito.perms import permutation_operator


class QuantumHedging:
    r"""Calculate optimal winning probabilities for hedging scenarios.

    Calculate the maximal and minimal winning probabilities for quantum
    hedging to occur in certain two-party scenarios [@arunachalam2017quantum;@molina2012hedging].

    Examples:
        This example illustrates the initial example of perfect hedging when Alice
        and Bob play two repetitions of the game where Alice prepares the maximally
        entangled state:

        \[
            u = \frac{1}{\sqrt{2}}|00\rangle + \frac{1}{\sqrt{2}}|11\rangle,
        \]

        and Alice applies the measurement operator defined by vector

        \[
            v = \cos(\pi/8)|00\rangle + \sin(\pi/8)|11\rangle.
        \]

        As was illustrated in [@molina2012hedging], the hedging value of the above scenario is
        \(\cos(\pi/8)^2 \approx 0.8536\). The primal and dual programs for the maximal probability agree on this
        value, and so do the two programs for the minimal probability.

        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.states import basis
        from numpy import kron, cos, sin, pi, sqrt
        from toqito.nonlocal_games.quantum_hedging import QuantumHedging

        e_0, e_1 = basis(2, 0), basis(2, 1)
        e_00, e_01 = kron(e_0, e_0), kron(e_0, e_1)
        e_10, e_11 = kron(e_1, e_0), kron(e_1, e_1)

        alpha = 1 / sqrt(2)
        theta = pi / 8
        w_var = alpha * cos(theta) * e_00 + sqrt(1 - alpha ** 2) * sin(theta) * e_11

        l_1 = -alpha * sin(theta) * e_00 + sqrt(1 - alpha ** 2) * cos(theta) * e_11
        l_2 = alpha * sin(theta) * e_10
        l_3 = sqrt(1 - alpha ** 2) * cos(theta) * e_01

        q_1 = w_var @ w_var.conj().T
        q_0 = l_1 @ l_1.conj().T + l_2 @ l_2.conj().T + l_3 @ l_3.conj().T

        molina_watrous = QuantumHedging(q_0, 1)

        # cos(pi/8)**2 \approx 0.8536
        print(np.around(molina_watrous.max_prob_outcome_a_primal(), decimals=2))
        print(np.around(molina_watrous.max_prob_outcome_a_dual(), decimals=2))
        print(np.around(molina_watrous.min_prob_outcome_a_primal(), decimals=2))
        print(np.around(molina_watrous.min_prob_outcome_a_dual(), decimals=2))
        ```

    """

    def __init__(self, q_a: np.ndarray, num_reps: int) -> None:
        """Initialize the variables for semidefinite program.

        Args:
            q_a: The fixed SDP variable.
            num_reps: The number of parallel repetitions.

        """
        self._q_a = q_a
        self._num_reps = num_reps

        # The 2n qubit systems are ordered Y_1, X_1, ..., Y_n, X_n; the partial trace is over the Y systems.
        self._sys = list(range(0, 2 * num_reps, 2))
        self._dim = [2] * (2 * num_reps)

        # For the dual problem, the following unitary operator is used to
        # permute the subsystems of Alice and Bob which is defined by the
        # action:
        #   π(y1 ⊗ y2 ⊗ x1 ⊗ x2) = y1 ⊗ x1 ⊗ y2 ⊗ x2
        # for all y1 ∈ Y1, y2 ∈ Y2, x1 ∈ X1, x2 ∈ X2.).
        perm = [system for pair in zip(range(num_reps), range(num_reps, 2 * num_reps)) for system in pair]
        self._pperm = permutation_operator(2, perm)

    def _primal(self, maximize: bool) -> float:
        """Solve the primal hedging program, maximizing or minimizing the probability of outcome "a"."""
        x_var = cvxpy.Variable((4**self._num_reps, 4**self._num_reps), hermitian=True)
        objective = cvxpy.real(cvxpy.trace(self._q_a.conj().T @ x_var))
        constraints = [partial_trace(x_var, self._sys, self._dim) == np.identity(2**self._num_reps), x_var >> 0]
        problem = cvxpy.Problem(cvxpy.Maximize(objective) if maximize else cvxpy.Minimize(objective), constraints)
        return solve_sdp(problem)

    def _dual(self, maximize: bool) -> float:
        """Solve the dual hedging program for the maximal (`maximize`) or minimal probability of outcome "a"."""
        y_var = cvxpy.Variable((2**self._num_reps, 2**self._num_reps), hermitian=True)
        objective = cvxpy.real(cvxpy.trace(y_var))
        lifted = self._pperm @ cvxpy.kron(np.eye(2**self._num_reps), y_var) @ self._pperm.conj().T
        if maximize:
            problem = cvxpy.Problem(cvxpy.Minimize(objective), [lifted >> self._q_a])
        else:
            problem = cvxpy.Problem(cvxpy.Maximize(objective), [lifted << self._q_a])
        return solve_sdp(problem)

    def max_prob_outcome_a_primal(self) -> float:
        r"""Compute the maximal probability for calculating outcome "a".

        The primal problem for the maximal probability of "a" is given as:

        \[
            \begin{equation}
                \begin{aligned}
                    \text{maximize:} \quad & \langle Q_{a_1} \otimes \ldots
                                             \otimes Q_{a_n}, X \rangle \\
                \text{subject to:} \quad & \text{Tr}_{\mathcal{Y}_1 \otimes
                                            \ldots \otimes \mathcal{Y}_n}(X) =
                                            I_{\mathcal{X}_1 \otimes \ldots
                                            \otimes \mathcal{X}_n},\\
                                            & X \in \text{Pos}(\mathcal{Y}_1
                                            \otimes \mathcal{X}_1 \otimes \ldots
                                            \otimes \mathcal{Y}_n \otimes
                                            \mathcal{X}_n)
                \end{aligned}
            \end{equation}
        \]

        Returns:
            The optimal maximal probability for obtaining outcome "a".

        """
        return self._primal(maximize=True)

    def max_prob_outcome_a_dual(self) -> float:
        r"""Compute the maximal probability for calculating outcome "a".

        The dual problem for the maximal probability of "a" is given as:

        \[
            \begin{equation}
                \begin{aligned}
                    \text{minimize:} \quad & \text{Tr}(Y) \\
                    \text{subject to:} \quad & \pi \left(I_{\mathcal{Y}_1
                    \otimes \ldots \otimes \mathcal{Y}_n} \otimes Y \right)
                    \pi^* \geq Q_{a_1} \otimes \ldots \otimes Q_{a_n}, \\
                    & Y \in \text{Herm} \left(\mathcal{X} \otimes \ldots \otimes
                    \mathcal{X}_n \right)
                \end{aligned}
            \end{equation}
        \]

        Returns:
            The optimal maximal probability for obtaining outcome "a".

        """
        return self._dual(maximize=True)

    def min_prob_outcome_a_primal(self) -> float:
        r"""Compute the minimal probability for calculating outcome "a".

        The primal problem for the minimal probability of "a" is given as:

        \[
            \begin{equation}
                \begin{aligned}
                    \text{minimize:} \quad & \langle Q_{a_1} \otimes \ldots
                                             \otimes Q_{a_n}, X \rangle \\
                \text{subject to:} \quad & \text{Tr}_{\mathcal{Y}_1 \otimes
                                            \ldots \otimes \mathcal{Y}_n}(X) =
                                            I_{\mathcal{X}_1 \otimes \ldots
                                            \otimes \mathcal{X}_n},\\
                                            & X \in \text{Pos}(\mathcal{Y}_1
                                            \otimes \mathcal{X}_1 \otimes \ldots
                                            \otimes \mathcal{Y}_n \otimes
                                            \mathcal{X}_n)
                \end{aligned}
            \end{equation}
        \]

        Returns:
            The optimal minimal probability for obtaining outcome "a".

        """
        return self._primal(maximize=False)

    def min_prob_outcome_a_dual(self) -> float:
        r"""Compute the minimal probability for calculating outcome "a".

        The dual problem for the minimal probability of "a" is given as:

        \[
        \begin{equation}
            \begin{aligned}
                \text{maximize:} \quad & \text{Tr}(Y) \\
                \text{subject to:} \quad & \pi \left(I_{\mathcal{Y}_1
                \otimes \ldots \otimes \mathcal{Y}_n} \otimes Y \right)
                \pi^* \leq Q_{a_1} \otimes \ldots \otimes Q_{a_n}, \\
                & Y \in \text{Herm} \left(\mathcal{X} \otimes \ldots \otimes
                \mathcal{X}_n \right)
            \end{aligned}
        \end{equation}
        \]

        Returns:
            The optimal minimal probability for obtaining outcome "a".

        """
        return self._dual(maximize=False)
