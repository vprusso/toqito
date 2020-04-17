"""Calculates probability of distinguishing via PPT measurements."""
from typing import List
import cvxpy
import numpy as np

from toqito.maps.partial_transpose import partial_transpose


def ppt_distinguishability(
    states: List[np.ndarray], probs: List[float] = None
) -> float:
    r"""
    Compute probability of distinguishing a state via PPT measurements [5]_.

    Implements the semidefinite program (SDP) whose optimal value is equal to
    the maximum probability of perfectly distinguishing orthogonal maximally
    entangled states using any PPT measurement; a measurement whose operators
    are positive under partial transpose. This SDP was explicitly provided in
    [5]_.

    Specifically, the function implements the dual problem (as this is
    computationally more efficient) and is defined as:

    .. math::

        \begin{equation}
            \begin{aligned}
                \text{minimize:} \quad & \frac{1}{k} \text{Tr}(Y) \\
                \text{subject to:} \quad & Y \geq \text{T}_{\mathcal{A}}
                                          (\rho_j), \quad j = 1, \ldots, k, \\
                                         & Y \in \text{Herm}(\mathcal{A} \otimes
                                          \mathcal{B}).
            \end{aligned}
        \end{equation}

    Examples
    ==========

    Consider the following Bell states

    .. math::
        \begin{equation}
            |\psi_0 \rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}, \quad
            |\psi_1 \rangle = \frac{|01\rangle + |10\rangle}{\sqrt{2}}, \quad
            |\psi_2 \rangle = \frac{|01\rangle - |10\rangle}{\sqrt{2}}, \quad
            |\psi_3 \rangle = \frac{|00\rangle - |11\rangle}{\sqrt{2}}. \quad
        \end{equation}

    It was illustrated in [6]_ that for the following set of states:

    The PPT distinguishability of the following states

    .. math::
        \begin{equation}
            \rho_1^{(2)} = \psi_0 \otimes \psi_0, \quad
            \rho_2^{(2)} = \psi_1 \otimes \psi_1, \quad
        \end{equation}

    should yield :math:`7/8 ~ 0.875` as was proved in [6]_.

    >>> from toqito.states.states.bell import bell
    >>> from toqito.states.optimizations.ppt_distinguishability import ppt_distinguishability
    >>> # Bell vectors:
    >>> psi_0 = bell(0)
    >>> psi_1 = bell(2)
    >>> psi_2 = bell(3)
    >>> psi_3 = bell(1)
    >>>
    >>> # YYD vectors from [6]_.
    >>> x_1 = np.kron(psi_0, psi_0)
    >>> x_2 = np.kron(psi_1, psi_3)
    >>> x_3 = np.kron(psi_2, psi_3)
    >>> x_4 = np.kron(psi_3, psi_3)
    >>>
    >>> # YYD density matrices.
    >>> rho_1 = x_1 * x_1.conj().T
    >>> rho_2 = x_2 * x_2.conj().T
    >>> rho_3 = x_3 * x_3.conj().T
    >>> rho_4 = x_4 * x_4.conj().T
    >>>
    >>> states = [rho_1, rho_2, rho_3, rho_4]
    >>> probs = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    >>> res = ppt_distinguishability(states, probs)
    0.875

    References
    ==========
    .. [5] Cosentino, Alessandro.
        "Positive-partial-transpose-indistinguishable states via semidefinite
        programming."
        Physical Review A 87.1 (2013): 012321.
        https://arxiv.org/abs/1205.1031

    .. [6] Yu, Nengkun, Runyao Duan, and Mingsheng Ying.
        "Four locally indistinguishable ququad-ququad orthogonal
        maximally entangled states."
        Physical review letters 109.2 (2012): 020506.
        https://arxiv.org/abs/1107.3224

    :param states: A list of density operators (matrices) corresponding to
                   quantum states.
    :param probs: A list of probabilities where `probs[i]` corresponds to the
                  probability that `states[i]` is selected by Alice.
    :return: The optimal probability with which the states can be distinguished
             via PPT measurements.
    """
    # Assume that at least one state is provided.
    if states is None or states == []:
        raise ValueError("InvalidStates: There must be at least one state " "provided.")

    # Assume uniform probability if no specific distribution is given.
    if probs is None:
        probs = [1 / len(states)] * len(states)
    if not np.isclose(sum(probs), 1):
        raise ValueError("Invalid: Probabilities must sum to 1.")

    dim_x, dim_y = states[0].shape

    # The variable `states` is provided as a list of vectors. Transform them
    # into density matrices.
    if dim_y == 1:
        for i, state_ket in enumerate(states):
            states[i] = state_ket * state_ket.conj().T

    constraints = []
    y_var = cvxpy.Variable((dim_x, dim_x), hermitian=True)
    objective = 1 / len(states) * cvxpy.Minimize(cvxpy.trace(cvxpy.real(y_var)))

    dim = int(np.log2(dim_x))
    dim_list = [2] * int(np.log2(dim_x))
    sys_list = list(range(1, dim, 2))

    for i, _ in enumerate(states):
        constraints.append(
            cvxpy.real(y_var)
            >> partial_transpose(states[i], sys=sys_list, dim=dim_list)
        )

    problem = cvxpy.Problem(objective, constraints)
    sol_default = problem.solve()

    return sol_default
