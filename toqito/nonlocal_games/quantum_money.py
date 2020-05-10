"""Calculates success probability of counterfeiting quantum money."""
import cvxpy
import numpy as np

from toqito.state_ops import tensor
from toqito.channels import partial_trace_cvx
from toqito.perms import permutation_operator


class QuantumMoney:
    r"""
    Create quantum money-type objects.

    Test.
    """

    def __init__(self, q_a: np.ndarray, num_reps: int = 1) -> None:
        """
        Construct quantum money object.

        :param q_a:
        :param num_reps:
        """
        self._q_a = q_a
        self._num_reps = num_reps

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

        >>> from toqito.state_ops import tensor
        >>> from toqito.nonlocal_games.quantum_money import QuantumMoney
        >>> from toqito.states import basis
        >>> import numpy as np
        >>> e_0, e_1 = basis(2, 0), basis(2, 1)
        >>> e_p = (e_0 + e_1) / np.sqrt(2)
        >>> e_m = (e_0 - e_1) / np.sqrt(2)
        >>>
        >>> e_000 = tensor(e_0, e_0, e_0)
        >>> e_111 = tensor(e_1, e_1, e_1)
        >>> e_ppp = tensor(e_p, e_p, e_p)
        >>> e_mmm = tensor(e_m, e_m, e_m)
        >>>
        >>> q_a = 1 / 4 * (e_000 * e_000.conj().T + e_111 * e_111.conj().T + \
        >>> e_ppp * e_ppp.conj().T + e_mmm * e_mmm.conj().T)
        >>> wiesner = QuantumMoney(q_a)
        >>> wiesner.counterfeit_attack(q_a)
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
        :return:
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
            constraints = [cvxpy.real(pperm @ kron_var @ pperm.conj().T) >> self._q_a]
        problem = cvxpy.Problem(objective, constraints)

        return problem.solve()

    def primal_problem(self, pperm: np.ndarray) -> float:
        """
        Primal problem for counterfeit attack.

        :param pperm:
        :return:
        """
        num_spaces = 3

        sys = list(range(1, num_spaces * self._num_reps))
        sys = [elem for elem in sys if elem % num_spaces != 0]

        # The dimension of each subsystem is assumed to be of dimension 2.
        dim = 2 * np.ones((1, num_spaces * self._num_reps)).astype(int).flatten()
        dim = dim.tolist()

        # Primal problem.
        x_var = cvxpy.Variable((8 ** self._num_reps, 8 ** self._num_reps), PSD=True)
        objective = cvxpy.Maximize(
            cvxpy.trace(pperm * self._q_a.conj().T * pperm.conj().T @ x_var)
        )
        constraints = [
            partial_trace_cvx(x_var, sys, dim) == np.identity(2 ** self._num_reps)
        ]
        problem = cvxpy.Problem(objective, constraints)

        return problem.solve()
