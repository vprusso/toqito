"""PPT distinguishability."""
from typing import List

import cvxpy
import numpy as np

from toqito.channels import partial_transpose
from .state_helper import __is_states_valid, __is_probs_valid


def ppt_distinguishability(
    states: List[np.ndarray], probs: List[float] = None
) -> float:
    r"""
    Compute probability of distinguishing a state via PPT measurements [COS13]_.

    Implements the semidefinite program (SDP) whose optimal value is equal to the maximum
    probability of perfectly distinguishing orthogonal maximally entangled states using any PPT
    measurement; a measurement whose operators are positive under partial transpose. This SDP was
    explicitly provided in [COS13]_.

    Specifically, the function implements the dual problem (as this is computationally more
    efficient) and is defined as:

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

    Examples
    ==========

    Consider the following Bell states

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

    :return: The optimal probability with which the states can be distinguished
             via PPT measurements.
    """
    constraints = []
    meas = []

    __is_states_valid(states)
    if probs is None:
        probs = [1 / len(states)] * len(states)
    __is_probs_valid(probs)

    dim_x, dim_y = states[0].shape

    # The variable `states` is provided as a list of vectors. Transform them
    # into density matrices.
    if dim_y == 1:
        for i, state_ket in enumerate(states):
            states[i] = state_ket * state_ket.conj().T

    y_var = cvxpy.Variable((dim_x, dim_x), hermitian=True)
    objective = cvxpy.Minimize(cvxpy.trace(cvxpy.real(y_var)))

    dim = int(np.log2(dim_x))
    dim_list = [2] * int(np.log2(dim_x))

    sys_list = list(range(1, dim, 2))
    if not sys_list:
        sys_list = [2]

    for i, _ in enumerate(states):
        meas.append(cvxpy.Variable((dim_x, dim_x), PSD=True))
        constraints.append(
            cvxpy.real(y_var - probs[i] * states[i])
            >> partial_transpose(meas[i], sys=sys_list, dim=dim_list)
        )

    problem = cvxpy.Problem(objective, constraints)
    sol_default = problem.solve()

    return sol_default
