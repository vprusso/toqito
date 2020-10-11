"""Calculates success probability of approximately cloning a quantum state."""
from typing import List, Union
import cvxpy
import numpy as np

from toqito.matrix_ops import tensor
from toqito.channels import partial_trace
from toqito.perms import permutation_operator


def optimal_clone(
    states: List[np.ndarray],
    probs: List[float],
    num_reps: int = 1,
    strategy: bool = False,
) -> Union[float, np.ndarray]:
    r"""
    Compute probability of counterfeiting quantum money [MVW12]_.

    The primal problem for the :math:`n`-fold parallel repetition is given as follows:

    .. math::
        \begin{equation}
            \begin{aligned}
                \text{maximize:} \quad &
                \langle W_{\pi} \left(Q^{\otimes n} \right) W_{\pi}^*, X \rangle \\
                \text{subject to:} \quad & \text{Tr}_{\mathcal{Y}^{\otimes n}
                                           \otimes \mathcal{Z}^{\otimes n}}(X)
                                           = \mathbb{I}_{\mathcal{X}^{\otimes
                                           n}},\\
                                           & X \in \text{Pos}(
                                           \mathcal{Y}^{\otimes n}
                                           \otimes \mathcal{Z}^{\otimes n}
                                           \otimes \mathcal{X}^{\otimes n}).
            \end{aligned}
        \end{equation}

    The dual problem for the :math:`n`-fold parallel repetition is given as follows:

    .. math::
        \begin{equation}
            \begin{aligned}
                \text{minimize:} \quad & \text{Tr}(Y) \\
                \text{subject to:} \quad & \mathbb{I}_{\mathcal{Y}^{\otimes n}
                \otimes \mathcal{Z}^{\otimes n}} \otimes Y \geq W_{\pi}
                \left( Q^{\otimes n} \right) W_{\pi}^*, \\
                & Y \in \text{Herm} \left(\mathcal{X}^{\otimes n} \right).
            \end{aligned}
        \end{equation}

    Examples
    ==========

    Wiesner's original quantum money scheme [Wies83]_ was shown in [MVW12]_ to have an optimal
    probability of 3/4 for succeeding a counterfeiting attack.

    Specifically, in the single-qubit case, Wiesner's quantum money scheme corresponds to the
    following ensemble:

    .. math::
        \left\{
            \left( \frac{1}{4}, |0\rangle \right),
            \left( \frac{1}{4}, |1\rangle \right),
            \left( \frac{1}{4}, |+\rangle \right),
            \left( \frac{1}{4}, |-\rangle \right)
        \right\},

    which yields the operator

    .. math::
        \begin{equation}
            Q = \frac{1}{4} \left(|000 \rangle \langle 000| + |111 \rangle \langle 111| +
                                  |+++ \rangle + \langle +++| + |--- \rangle \langle ---| \right).
        \end{equation}

    We can see that the optimal value we obtain in solving the SDP is 3/4.

    >>> from toqito.state_opt import optimal_clone
    >>> from toqito.states import basis
    >>> import numpy as np
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> e_p = (e_0 + e_1) / np.sqrt(2)
    >>> e_m = (e_0 - e_1) / np.sqrt(2)
    >>>
    >>> states = [e_0, e_1, e_p, e_m]
    >>> probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    >>> wiesner = optimal_clone(states, probs)
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

    :param states: A list of states provided as either matrices or vectors.
    :param probs: Respective list of probabilities each state is selected.
    :param num_reps: Number of parallel repetitions to perform.
    :param strategy: Boolean that denotes whether to return strategy.
    :return: The optimal probability with of counterfeiting quantum money.
    """
    dim = len(states[0]) ** 3

    # Construct the following operator:
    #                                ___               ___
    # Q = ∑_{k=1}^N p_k |ψ_k ⊗ ψ_k ⊗ ψ_k> <ψ_k ⊗ ψ_k ⊗ ψ_k|
    q_a = np.zeros((dim, dim))
    for k, state in enumerate(states):
        q_a += (
            probs[k]
            * tensor(state, state, state.conj())
            * tensor(state, state, state.conj()).conj().T
        )

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
        q_a = tensor(q_a, num_reps)
        perm = []
        for i in range(1, num_spaces + 1):
            perm.append(i)
            var = i
            for j in range(1, num_reps):
                perm.append(var + num_spaces * j)
        pperm = permutation_operator(2, perm)

    if strategy:
        return primal_problem(q_a, pperm, num_reps)
    return dual_problem(q_a, pperm, num_reps)


def primal_problem(q_a: np.ndarray, pperm: np.ndarray, num_reps: int) -> float:
    """
    Primal problem for counterfeit attack.

    As the primal problem takes longer to solve than the dual problem (as
    the variables are of larger dimension), the primal problem is only here
    for reference.

    :return: The optimal value of performing a counterfeit attack.
    """
    num_spaces = 3

    sys = list(range(1, num_spaces * num_reps))
    sys = [elem for elem in sys if elem % num_spaces != 0]

    # The dimension of each subsystem is assumed to be of dimension 2.
    dim = 2 * np.ones((1, num_spaces * num_reps)).astype(int).flatten()
    dim = dim.tolist()

    x_var = cvxpy.Variable((8 ** num_reps, 8 ** num_reps), hermitian=True)
    if num_reps == 1:
        objective = cvxpy.Maximize(cvxpy.trace(cvxpy.real(q_a.conj().T @ x_var)))
    else:
        objective = cvxpy.Maximize(
            cvxpy.trace(cvxpy.real(pperm @ q_a.conj().T @ pperm.conj().T @ x_var))
        )
    constraints = [
        partial_trace(x_var, sys, dim) == np.identity(2 ** num_reps),
        x_var >> 0,
    ]
    problem = cvxpy.Problem(objective, constraints)

    return problem.solve()


def dual_problem(q_a: np.ndarray, pperm: np.ndarray, num_reps: int) -> float:
    """
    Dual problem for counterfeit attack.

    :return: The optimal value of performing a counterfeit attack.
    """
    y_var = cvxpy.Variable((2 ** num_reps, 2 ** num_reps), hermitian=True)
    objective = cvxpy.Minimize(cvxpy.trace(cvxpy.real(y_var)))

    kron_var = cvxpy.kron(cvxpy.kron(np.eye(2 ** num_reps), np.eye(2 ** num_reps)), y_var)

    if num_reps == 1:
        constraints = [cvxpy.real(kron_var) >> q_a]
    else:
        constraints = [cvxpy.real(kron_var) >> pperm @ q_a @ pperm.conj().T]
    problem = cvxpy.Problem(objective, constraints)

    return problem.solve()
