"""Calculates success probability of counterfeiting quantum money."""
from typing import List
import cvxpy
import numpy as np

from toqito.matrix_ops import tensor
from toqito.channels import partial_trace_cvx
from toqito.perms import permutation_operator


class QuantumMoney:
    r"""
    Create quantum money-type objects.

    Test.
    """

    def __init__(
        self, states: List[np.ndarray], probs: List[float], num_reps: int = 1
    ) -> None:
        """
        Construct quantum money object.

        :param states: The quantum states.
        :param probs: The probabilities with which to select the states.
        :param num_reps: The number of parallel repetitions.
        """
        self._num_reps = num_reps

        dim = len(states[0]) ** 3

        # Construct the following operator:
        #                              __              __
        # Q = ∑_{k=1}^N p_k |ψk ⊗ ψk ⊗ ψk}> <ψk ⊗ ψk ⊗ ψk|
        self._q_a = np.zeros((dim, dim))
        for k, state in enumerate(states):
            self._q_a += (
                probs[k]
                * tensor(state, state, state.conj())
                * tensor(state, state, state.conj()).conj().T
            )

    def counterfeit_attack(self) -> float:
        r"""
        Compute probability of counterfeiting quantum money [MVW12]_.

        The primal problem for the :math:`n`-fold parallel repetition is given as
        follows:

        .. math::
            \begin{equation}
                \begin{aligned}
                    \text{maximize:} \quad & \langle W_{\pi} \left(Q^{\otimes n}
                                             \right) W_{\pi}^*, X \rangle \\
                    \text{subject to:} \quad & \text{Tr}_{\mathcal{Y}^{\otimes n}
                                               \otimes \mathcal{Z}^{\otimes n}}(X)
                                               = \mathbb{I}_{\mathcal{X}^{\otimes
                                               n}},\\
                                               & X \in \text{Pos}(
                                               \mathcal{Y}^{\otimes n}
                                               \otimes \mathcal{Z}^{\otimes n}
                                               \otimes \mathcal{X}^{\otimes n})
                \end{aligned}
            \end{equation}

        The dual problem for the :math:`n`-fold parallel repetition is given as
        follows:

        .. math::
                \begin{equation}
                    \begin{aligned}
                        \text{minimize:} \quad & \text{Tr}(Y) \\
                        \text{subject to:} \quad & \mathbb{I}_{\mathcal{Y}^{\otimes n}
                        \otimes \mathcal{Z}^{\otimes n}} \otimes Y \geq W_{\pi}
                        \left( Q^{\otimes n} \right) W_{\pi}^*, \\
                        & Y \in \text{Herm} \left(\mathcal{X}^{\otimes n} \right)
                    \end{aligned}
                \end{equation}

        Examples
        ==========

        Wiesner's original quantum money scheme [Wies83]_ was shown in [MVW12]_ to
        have an optimal probability of 3/4 for succeeding a counterfeiting attack.

        Specifically, in the single-qubit case, Wiesner's quantum money scheme
        corresponds to the following ensemble:

        .. math::
            \left{
                \left( \frac{1}{4}, |0\rangle \right),
                \left( \frac{1}{4}, |1\rangle \right),
                \left( \frac{1}{4}, |+\rangle \right),
                \left( \frac{1}{4}, |-\rangle \right)
            \right},

        which yields the operator

        .. math::
            Q = \frac{1}{4} \left(
                |000\rangle + \langle 000| + |111\rangle \langle 111| +
                |+++\rangle + \langle +++| + |---\rangle \langle ---|
            \right)

        We can see that the optimal value we obtain in solving the SDP is 3/4.

        >>> from toqito.matrix_ops import tensor
        >>> from toqito.nonlocal_games.quantum_money import QuantumMoney
        >>> from toqito.states import basis
        >>> import numpy as np
        >>> e_0, e_1 = basis(2, 0), basis(2, 1)
        >>> e_p = (e_0 + e_1) / np.sqrt(2)
        >>> e_m = (e_0 - e_1) / np.sqrt(2)
        >>>
        >>> states = [e_0, e_1, e_p, e_m]
        >>> probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
        >>> wiesner = QuantumMoney(states, probs)
        >>> wiesner.counterfeit_attack()
        0.749999999967631

        References
        ==========
        .. [MVW12] Abel Molina, Thomas Vidick, and John Watrous.
            "Optimal counterfeiting attacks and generalizations for Wiesner’s
            quantum money."
            Conference on Quantum Computation, Communication, and Cryptography.
            Springer, Berlin, Heidelberg, 2012.
            https://arxiv.org/abs/1202.4010

        .. [Wies83] Stephen Wiesner
            "Conjugate coding."
            ACM Sigact News 15.1 (1983): 78-88.
            https://dl.acm.org/doi/pdf/10.1145/1008908.1008920

        :return: The optimal probability with of counterfeiting quantum money.
        """
        # The system is over:
        # Y_1 ⊗ Z_1 ⊗ X_1, ... , Y_n ⊗ Z_n ⊗ X_n.
        num_spaces = 3

        # In the event of more than a single repetition, one needs to apply a
        # permutation operator to the variables in the SDP to properly align
        # the spaces.
        if self._num_reps == 1:
            pperm = np.array([1])
        else:
            # The permutation vector `perm` contains elements of the
            # sequence from: https://oeis.org/A023123
            self._q_a = tensor(self._q_a, self._num_reps)
            perm = []
            for i in range(1, num_spaces + 1):
                perm.append(i)
                var = i
                for j in range(1, self._num_reps):
                    perm.append(var + num_spaces * j)
            pperm = permutation_operator(2, perm)

        return self.dual_problem(pperm)

    def dual_problem(self, pperm: np.ndarray) -> float:
        """
        Dual problem for counterfeit attack.

        :param pperm:
        :return: The optimal value of performing a counterfeit attack.
        """
        y_var = cvxpy.Variable(
            (2 ** self._num_reps, 2 ** self._num_reps), hermitian=True
        )
        objective = cvxpy.Minimize(cvxpy.trace(cvxpy.real(y_var)))

        kron_var = cvxpy.kron(
            cvxpy.kron(np.eye(2 ** self._num_reps), np.eye(2 ** self._num_reps)), y_var
        )

        if self._num_reps == 1:
            constraints = [cvxpy.real(kron_var) >> self._q_a]
        else:
            constraints = [cvxpy.real(kron_var) >> pperm @ self._q_a @ pperm.conj().T]
        problem = cvxpy.Problem(objective, constraints)

        return problem.solve()

    def primal_problem(self, pperm: np.ndarray) -> float:
        """
        Primal problem for counterfeit attack.

        As the primal problem takes longer to solve than the dual problem (as
        the variables are of larger dimension), the primal problem is only here
        for reference.

        :param pperm:
        :return: The optimal value of performing a counterfeit attack.
        """
        num_spaces = 3

        sys = list(range(1, num_spaces * self._num_reps))
        sys = [elem for elem in sys if elem % num_spaces != 0]

        # The dimension of each subsystem is assumed to be of dimension 2.
        dim = 2 * np.ones((1, num_spaces * self._num_reps)).astype(int).flatten()
        dim = dim.tolist()

        # Primal problem.
        x_var = cvxpy.Variable(
            (8 ** self._num_reps, 8 ** self._num_reps), hermitian=True
        )
        objective = cvxpy.Maximize(
            cvxpy.trace(cvxpy.real(pperm @ self._q_a.conj().T @ pperm.conj().T @ x_var))
        )
        constraints = [
            partial_trace_cvx(x_var, sys, dim) == np.identity(2 ** self._num_reps),
            x_var >> 0,
        ]
        problem = cvxpy.Problem(objective, constraints)

        return problem.solve()
