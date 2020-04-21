"""Calculates success probability of counterfeiting quantum money."""
import cvxpy
import numpy as np

from toqito.linear_algebra.operations.tensor import tensor_n
from toqito.perms.permutation_operator import permutation_operator


def counterfeit_attack(q_a: np.ndarray, num_reps: int = 1) -> float:
    r"""
    Compute probability of counterfeiting quantum money [MVW12]_.

    The primal problem for the :math:`n`-fold parallel repetition is given as
    follows:

    .. math::
        \begin{equation}
            \begin{aligned}
                \text{maximize:} \quad & \langle W_{\pi} \left(
                                         Q^{\otimes n} \right) W_{\pi}^*, X
                                         \rangle \\
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

    >>> from toqito.linear_algebra.operations.tensor import tensor_list
    >>> from toqito.core.ket import ket
    >>> import numpy as np
    >>> e_0, e_1 = ket(2, 0), ket(2, 1)
    >>> e_p = (e_0 + e_1) / np.sqrt(2)
    >>> e_m = (e_0 - e_1) / np.sqrt(2)
    >>>
    >>> e_000 = tensor_list([e_0, e_0, e_0])
    >>> e_111 = tensor_list([e_1, e_1, e_1])
    >>> e_ppp = tensor_list([e_p, e_p, e_p])
    >>> e_mmm = tensor_list([e_m, e_m, e_m])
    >>>
    >>> q_a = 1 / 4 * (e_000 * e_000.conj().T + e_111 * e_111.conj().T + \
    >>> e_ppp * e_ppp.conj().T + e_mmm * e_mmm.conj().T)
    >>> print(counterfeit_attack(q_a))
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

    :param q_a: The fixed SDP variable.
    :param num_reps: The number of parallel repetitions.
    :return: The optimal probability with of counterfeiting quantum money.
    """

    # The system is over:
    # Y_1 ⊗ Z_1 ⊗ X_1, ... , Y_n ⊗ Z_n ⊗ X_n.
    num_spaces = 3

    # In the event of more than a single repetition, one needs to apply a
    # permutation operator to the variables in the SDP to properly align
    # the spaces.
    if num_reps == 1:
        pperm = np.array([1])
    else:
        # The permutation vector `perm` contains elements of the
        # sequence from: https://oeis.org/A023123
        q_a = tensor_n(q_a, num_reps)
        perm = []
        for i in range(1, num_spaces + 1):
            perm.append(i)
            var = i
            for j in range(1, num_reps):
                perm.append(var + num_spaces * j)
        pperm = permutation_operator(2, perm)

    return dual_problem(q_a, pperm, num_reps)


def dual_problem(q_a: np.ndarray, pperm: np.ndarray, num_reps: int) -> float:
    """
    :param q_a:
    :param pperm:
    :param num_reps:
    :return:
    """
    y_var = cvxpy.Variable((2 ** num_reps, 2 ** num_reps), hermitian=True)
    objective = cvxpy.Minimize(cvxpy.trace(cvxpy.real(y_var)))

    kron_var = cvxpy.kron(
        cvxpy.kron(np.eye(2 ** num_reps), np.eye(2 ** num_reps)), y_var
    )

    if num_reps == 1:
        constraints = [cvxpy.real(kron_var) >> q_a]
    else:
        constraints = [cvxpy.real(pperm @ kron_var @ pperm.conj().T) >> q_a]
    problem = cvxpy.Problem(objective, constraints)

    return problem.solve()


# def primal_problem(q_a: np.ndarray, pperm: np.ndarray, num_reps: int) -> float:
#     num_spaces = 3
#
#     sys = list(range(1, num_spaces * num_reps))
#     sys = [elem for elem in sys if elem % num_spaces != 0]
#
#     # The dimension of each subsystem is assumed to be of dimension 2.
#     dim = 2 * np.ones((1, num_spaces * num_reps)).astype(int).flatten()
#     dim = dim.tolist()
#
#     # Primal problem.
#     x_var = cvxpy.Variable((8 ** num_reps, 8 ** num_reps), PSD=True)
#     objective = cvxpy.Maximize(
#         cvxpy.trace(pperm * q_a.conj().T * pperm.conj().T @ x_var)
#     )
#     constraints = [partial_trace_cvx(x_var, sys, dim) == np.identity(2 ** num_reps)]
#     problem = cvxpy.Problem(objective, constraints)
#
#     return problem.solve()
