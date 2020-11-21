"""PPT distinguishability."""
from typing import List

import cvxpy
import numpy as np

from toqito.channels import partial_transpose
from .state_helper import __is_states_valid, __is_probs_valid


def ppt_distinguishability(
    states: List[np.ndarray],
    probs: List[float] = None,
    dist_method="min-error",
    strategy=False,
) -> float:
    r"""
    Compute probability of optimally distinguishing a state via PPT measurements [COS13]_.

    Implements the semidefinite program (SDP) whose optimal value is equal to the maximum
    probability of perfectly distinguishing orthogonal maximally entangled states using any PPT
    measurement; a measurement whose operators are positive under partial transpose. This SDP was
    explicitly provided in [COS13]_.

    One can specify the distinguishability method using the :code:`dist_method` argument.

    For :code:`dist_method = "min-error"`, this is the default method that yields the probability of
    distinguishing quantum states via PPT measurements that minimize the probability of error.

    For :code:`dist_method = "unambiguous"`, Alice and Bob never provide an incorrect answer,
    although it is possible that their answer is inconclusive.

    Examples
    ==========

    Consider the following Bell states:

    .. math::
        \begin{equation}
            \begin{aligned}
            |\psi_0 \rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}, &\quad
            |\psi_1 \rangle = \frac{|01\rangle + |10\rangle}{\sqrt{2}}, \\
            |\psi_2 \rangle = \frac{|01\rangle - |10\rangle}{\sqrt{2}}, &\quad
            |\psi_3 \rangle = \frac{|00\rangle - |11\rangle}{\sqrt{2}}.
            \end{aligned}
        \end{equation}

    It was illustrated in [YDY12]_ that for the following set of states

    .. math::
        \begin{equation}
            \begin{aligned}
            \rho_1^{(2)} &= |\psi_0 \rangle | \psi_0 \rangle \langle \psi_0 | \langle \psi_0 |, \\
            \rho_2^{(2)} &= |\psi_1 \rangle | \psi_3 \rangle \langle \psi_1 | \langle \psi_3 |, \\
            \rho_3^{(2)} &= |\psi_2 \rangle | \psi_3 \rangle \langle \psi_2 | \langle \psi_3 |, \\
            \rho_4^{(2)} &= |\psi_3 \rangle | \psi_3 \rangle \langle \psi_3 | \langle \psi_3 |, \\
            \end{aligned}
        \end{equation}

    that the optimal probability of distinguishing via a PPT measurement should yield
    :math:`7/8 \approx 0.875` as was proved in [YDY12]_.

    >>> from toqito.states import bell
    >>> from toqito.state_opt import ppt_distinguishability
    >>> # Bell vectors:
    >>> psi_0 = bell(0)
    >>> psi_1 = bell(2)
    >>> psi_2 = bell(3)
    >>> psi_3 = bell(1)
    >>>
    >>> # YDY vectors from [YDY12]_.
    >>> x_1 = np.kron(psi_0, psi_0)
    >>> x_2 = np.kron(psi_1, psi_3)
    >>> x_3 = np.kron(psi_2, psi_3)
    >>> x_4 = np.kron(psi_3, psi_3)
    >>>
    >>> # YDY density matrices.
    >>> rho_1 = x_1 * x_1.conj().T
    >>> rho_2 = x_2 * x_2.conj().T
    >>> rho_3 = x_3 * x_3.conj().T
    >>> rho_4 = x_4 * x_4.conj().T
    >>>
    >>> states = [rho_1, rho_2, rho_3, rho_4]
    >>> probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    >>> ppt_distinguishability(states, probs)
    0.875

    References
    ==========
    .. [COS13] Cosentino, Alessandro.
        "Positive-partial-transpose-indistinguishable states via semidefinite
        programming."
        Physical Review A 87.1 (2013): 012321.
        https://arxiv.org/abs/1205.1031

    .. [YDY12] Yu, Nengkun, Runyao Duan, and Mingsheng Ying.
        "Four locally indistinguishable ququad-ququad orthogonal
        maximally entangled states."
        Physical review letters 109.2 (2012): 020506.
        https://arxiv.org/abs/1107.3224

    :param states: A list of states provided as either matrices or vectors.
    :param probs: Respective list of probabilities each state is selected.
    :param dist_method: Method of distinguishing to use.
    :param strategy: Returns strategy if :code:`True` and does not otherwise.
    :return: The optimal probability with which the states can be distinguished
             via PPT measurements.
    """
    __is_states_valid(states)
    if probs is None:
        probs = [1 / len(states)] * len(states)
    __is_probs_valid(probs)

    _, dim_y = states[0].shape

    # The variable `states` is provided as a list of vectors. Transform them
    # into density matrices.
    if dim_y == 1:
        for i, state_ket in enumerate(states):
            states[i] = state_ket * state_ket.conj().T

    if strategy:
        return primal_problem(states, probs, dist_method)
    return dual_problem(states, probs, dist_method)


def primal_problem(
    states: List[np.ndarray], probs: List[float] = None, dist_method="min-error"
) -> float:
    r"""
    Calculate primal problem for PPT distinguishability.

    The minimum-error semidefinite program implemented is defined as:

    .. math::
    \begin{equation}
        \begin{aligned}
            \text{maximize:} \quad & \sum_{j=1}^k \langle P_j, \rho_j \rangle \\
            \text{subject to:} \quad & P_1 + \cdots + P_k = \mathbb{I}_{\mathcal{A}}
                                        \otimes \mathbb{I}_{\mathcal{B}}, \\
                                     & P_1, \ldots, P_k \in \text{PPT}(\mathcal{A} : \mathcal{B}).
        \end{aligned}
    \end{equation}

    The unambiguous semidefinite program implemented is defined as:

    .. math::
    \begin{equation}
        \begin{aligned}
            \text{maximize:} \quad & \sum_{j=1}^k \langle P_j, \rho_j \rangle \\
            \text{subject to:} \quad & P_1 + \cdots + P_k = \mathbb{I}_{\mathcal{A}}
                                        \otimes \mathbb{I}_{\mathcal{B}}, \\
                                     & P_1, \ldots, P_k
                                      \in \text{PPT}(\mathcal{A} : \mathcal{B}), \\
                                     & \langle P_i, \rho_j \rangle = 0,
                                       \quad 1 \leq i, j \leq k, \quad i \not= j.
        \end{aligned}
    \end{equation}

    :param states: A list of states provided as either matrices or vectors.
    :param probs: Respective list of probabilities each state is selected.
    :param dist_method: Method of distinguishing to use.
    :return: The optimal value of the PPT primal problem SDP.
    """
    dim_x, _ = states[0].shape

    obj_func = []
    meas = []
    constraints = []

    dim = int(np.log2(dim_x))
    dim_list = [2] * int(np.log2(dim_x))

    sys_list = list(range(1, dim, 2))

    # Unambiguous consists of k + 1 operators, where the outcome of the k+1^st corresponds to the
    # inconclusive answer.
    if dist_method == "unambiguous":
        for i in range(len(states) + 1):
            meas.append(cvxpy.Variable((dim_x, dim_x), PSD=True))
            constraints.append(partial_transpose(meas[i], sys_list, dim_list) >> 0)

        for i, _ in enumerate(states):
            for j, _ in enumerate(states):
                if i != j:
                    constraints.append(probs[j] * cvxpy.trace(states[j].conj().T @ meas[i]) == 0)

    # Minimize error of distinguishing via PPT measurements.
    elif dist_method == "min-error":
        for i, _ in enumerate(states):
            meas.append(cvxpy.Variable((dim_x, dim_x), PSD=True))
            constraints.append(partial_transpose(meas[i], sys_list, dim_list) >> 0)

    for i, _ in enumerate(states):
        obj_func.append(probs[i] * cvxpy.trace(states[i].conj().T @ meas[i]))

    constraints.append(sum(meas) == np.identity(dim_x))

    objective = cvxpy.Maximize(sum(obj_func))
    problem = cvxpy.Problem(objective, constraints)
    sol_default = problem.solve()

    return sol_default


def dual_problem(
    states: List[np.ndarray], probs: List[float] = None, dist_method="min-error"
) -> float:
    r"""
    Calculate dual problem for PPT distinguishability.

    The minimum-error semidefinite program implemented is defined as:

    .. math::
    \begin{equation}
        \begin{aligned}
            \text{minimize:} \quad & \frac{1}{k} \text{Tr}(Y) \\
            \text{subject to:} \quad & Y - \rho_j \geq \text{T}_{\mathcal{A}} (Q_j),
                                        \quad j = 1, \ldots, k, \\
                                     & Y \in \text{Herm}(\mathcal{A} \otimes
                                        \mathcal{B}), \\
                                     & Q_1, \ldots, Q_k \in
                                        \text{Pos}(\mathcal{A} \otimes \mathcal{B}).
        \end{aligned}
    \end{equation}

    The unambiguous semidefinite program implemented is defined as:

    .. math::
    \begin{equation}
        \begin{aligned}
            \text{minimize:} \quad & \frac{1}{k} \text{Tr}(Y) \\
            \text{subject to:} \quad & Y - \rho_j + \sum_{\substack{i \leq i \leq k \\ i \not= j}}
                                        y_{i,j} \rho_i \geq T_{\mathcal{A}}(Q_j),
                                        \quad j = 1, \ldots, k, \\
                                     & Y \geq T_{\mathcal{A}}(Q_{k+1}), \\
                                     & Y \in \text{Herm}(\mathcal{A} \otimes
                                        \mathcal{B}), \\
                                     & Q_1, \ldots, Q_k \in
                                        \text{Pos}(\mathcal{A} \otimes \mathcal{B}), \\
                                     & y_{i,j} \in \mathcal{R}. \quad 1 \leq i, j \leq k,
                                        \quad i \not= j.
        \end{aligned}
    \end{equation}

    :param states: A list of states provided as either matrices or vectors.
    :param probs: Respective list of probabilities each state is selected.
    :param dist_method: Method of distinguishing to use.
    :return: The optimal value of the PPT dual problem SDP.
    """
    constraints = []
    meas = []

    dim_x, _ = states[0].shape

    y_var = cvxpy.Variable((dim_x, dim_x), hermitian=True)
    objective = cvxpy.Minimize(cvxpy.trace(cvxpy.real(y_var)))

    dim = int(np.log2(dim_x))
    dim_list = [2] * int(np.log2(dim_x))
    sys_list = list(range(1, dim, 2))
    # dim_list = [3, 3]

    if dist_method == "min-error":
        for i, _ in enumerate(states):
            meas.append(cvxpy.Variable((dim_x, dim_x), PSD=True))
            constraints.append(
                cvxpy.real(y_var - probs[i] * states[i])
                >> partial_transpose(meas[i], sys=sys_list, dim=dim_list)
            )

    if dist_method == "unambiguous":
        for j, _ in enumerate(states):
            sum_val = 0
            for i, _ in enumerate(states):
                if i != j:
                    sum_val += cvxpy.real(cvxpy.Variable()) * probs[i] * states[i]
            meas.append(cvxpy.Variable((dim_x, dim_x), PSD=True))
            constraints.append(
                cvxpy.real(y_var - probs[j] * states[j] + sum_val)
                >> partial_transpose(meas[j], sys=sys_list, dim=dim_list)
            )

        meas.append(cvxpy.Variable((dim_x, dim_x), PSD=True))
        constraints.append(
            cvxpy.real(y_var) >> partial_transpose(meas[-1], sys=sys_list, dim=dim_list)
        )

    problem = cvxpy.Problem(objective, constraints)
    sol_default = problem.solve()

    # print(np.around(y_var.value, decimals=3))

    return sol_default
