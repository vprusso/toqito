"""State distinguishability."""
from typing import List

import cvxpy
import numpy as np
from .state_helper import __is_states_valid, __is_probs_valid


def state_distinguishability(
    states: List[np.ndarray], probs: List[float] = None, dist_method: str = "min-error"
) -> float:
    r"""
    Compute probability of state distinguishability [ELD03]_.

    The "quantum state distinguishability" problem involves a collection of :math:`n` quantum states

    .. math::
        \rho = \{ \rho_0, \ldots, \rho_n \},

    as well as a list of corresponding probabilities

    .. math::
        p = \{ p_0, \ldots, p_n \}.

    Alice chooses :math:`i` with probability :math:`p_i` and creates the state :math:`\rho_i` Bob
    wants to guess which state he was given from the collection of states.

    One can specify the distinguishability method using the :code:`dist_method` argument.

    For :code:`dist_method = "min-error"`, this is the default method that yields the probability of
    distinguishing quantum states that minimize the probability of error.

    For :code:`dist_method = "unambiguous"`, Alice and Bob never provide an incorrect answer,
    although it is possible that their answer is inconclusive.

    When :code:`dist_method = "min-error"`, this function implements the following semidefinite
    program that provides the optimal probability with which Bob can conduct quantum state
    distinguishability.

    .. math::
        \begin{align*}
            \text{maximize:} \quad & \sum_{i=0}^n p_i \langle M_i, \rho_i \rangle \\
            \text{subject to:} \quad & M_0 + \ldots + M_n = \mathbb{I},\\
                                     & M_0, \ldots, M_n \geq 0
        \end{align*}

    When :code:`dist_method = "unambiguous"`, this function implements the following semidefinite
    program that provides the optimal probability with which Bob can conduct unambiguous quantum
    state distinguishability.

    .. math::
        \begin{align*}
            \text{maximize:} \quad & \sum_{i=0}^n p_i \langle M_i, \rho_i \rangle \\
            \text{subject to:} \quad & M_0 + \ldots + M_{n+1} = \mathbb{I},\\
                                     & \langle M_i, \rho_j \rangle = 0,
                                       \quad 1 \leq i, j \leq n, \quad i \not= j.
                                     & M_0, \ldots, M_n \geq 0
        \end{align*}

    Examples
    ==========

    State distinguishability for two state density matrices.

    >>> from toqito.states import basis, bell
    >>> from toqito.state_opt import state_distinguishability
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> e_00 = e_0 * e_0.conj().T
    >>> e_11 = e_1 * e_1.conj().T
    >>> states = [e_00, e_11]
    >>> probs = [1 / 2, 1 / 2]
    >>> res = state_distinguishability(states, probs)
    0.5000000000006083

    References
    ==========
    .. [ELD03] Eldar, Yonina C.
        "A semidefinite programming approach to optimal unambiguous
        discrimination of quantum states."
        IEEE Transactions on information theory 49.2 (2003): 446-456.
        https://arxiv.org/abs/quant-ph/0206093

    :param states: A list of states provided as either matrices or vectors.
    :param probs: Respective list of probabilities each state is selected.
    :param dist_method: Method of distinguishing to use.
    :return: The optimal probability with which Bob can distinguish the state.
    """
    obj_func = []
    measurements = []
    constraints = []

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

    # Unambiguous state discrimination has an additional constraint on the states and measurements.
    if dist_method == "unambiguous":

        # Note we have one additional measurement operator in the unambiguous case.
        for i in range(len(states) + 1):
            measurements.append(cvxpy.Variable((dim_x, dim_x), PSD=True))

        # This is an extra condition required for the unambiguous case.
        for i, _ in enumerate(states):
            for j, _ in enumerate(states):
                if i != j:
                    constraints.append(cvxpy.trace(states[j].conj().T @ measurements[i]) == 0)

    if dist_method == "min-error":
        for i, _ in enumerate(states):
            measurements.append(cvxpy.Variable((dim_x, dim_x), PSD=True))

    # Objective function is the inner product between the states and measurements.
    for i, _ in enumerate(states):
        obj_func.append(probs[i] * cvxpy.trace(states[i].conj().T @ measurements[i]))

    constraints.append(sum(measurements) == np.identity(dim_x))

    objective = cvxpy.Maximize(sum(obj_func))
    problem = cvxpy.Problem(objective, constraints)
    sol_default = problem.solve()

    return sol_default
