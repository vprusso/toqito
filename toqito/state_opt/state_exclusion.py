"""State exclusion."""
from typing import List

import cvxpy
import numpy as np

from .state_helper import __is_states_valid, __is_probs_valid


def state_exclusion(
    states: List[np.ndarray], probs: List[float] = None, method: str = "conclusive"
) -> float:
    r"""
    Compute probability of single state exclusion.

    The *quantum state exclusion* problem involves a collection of :math:`n`
    quantum states

    .. math::
        \rho = \{ \rho_0, \ldots, \rho_n \},

    as well as a list of corresponding probabilities

    .. math::
        p = \{ p_0, \ldots, p_n \}.

    Alice chooses :math:`i` with probability :math:`p_i` and creates the state
    :math:`\rho_i`.

    Bob wants to guess which state he was *not* given from the collection of
    states. State exclusion implies that ability to discard (with certainty) at
    least one out of the "n" possible quantum states by applying a measurement.

    This function implements the following semidefinite program that provides
    the optimal probability with which Bob can conduct quantum state exclusion.

        .. math::
            \begin{equation}
                \begin{aligned}
                    \text{minimize:} \quad & \sum_{i=0}^n p_i \langle M_i,
                                                \rho_i \rangle \\
                    \text{subject to:} \quad & M_0 + \ldots + M_n =
                                               \mathbb{I}, \\
                                             & M_0, \ldots, M_n >= 0.
                \end{aligned}
            \end{equation}

    The conclusive state exclusion SDP is written explicitly in [BJOP14]_. The problem of conclusive
    state exclusion was also thought about under a different guise in [PBR12]_.


    For the unambiguous case, the following semidefinite program that provides
    the optimal probability with which Bob can conduct quantum state exclusion.

    .. math::
        \begin{align*}
            \text{maximize:} \quad & \sum_{i=0}^n \sum_{j=0}^n
                                     \langle M_i, \rho_j \rangle \\
            \text{subject to:} \quad & \sum_{i=0}^n M_i \leq \mathbb{I},\\
                                     & \text{Tr}(\rho_i M_i) = 0,
                                       \quad \quad \forall 1  \leq i \leq n, \\
                                     & M_0, \ldots, M_n \geq 0
        \end{align*}

    Examples
    ==========

    Consider the following two Bell states

    .. math::
        \begin{equation}
            \begin{aligned}
                u_0 &= \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right), \\
                u_1 &= \frac{1}{\sqrt{2}} \left( |00 \rangle - |11 \rangle \right).
            \end{aligned}
        \end{equation}

    For the corresponding density matrices :math:`\rho_0 = u_0 u_0^*` and
    :math:`\rho_1 = u_1 u_1^*`, we may construct a set

    .. math::
        \rho = \{\rho_0, \rho_1 \}

    such that

    .. math::
        p = \{1/2, 1/2\}.

    It is not possible to conclusively exclude either of the two states. We can see that the result
    of the function in :code:`toqito` yields a value of :math:`0` as the probability for this to
    occur.

    >>> from toqito.state_opt import state_exclusion
    >>> from toqito.states import bell
    >>> import numpy as np
    >>> rho1 = bell(0) * bell(0).conj().T
    >>> rho2 = bell(1) * bell(1).conj().T
    >>>
    >>> states = [rho1, rho2]
    >>> probs = [1/2, 1/2]
    >>>
    >>> state_exclusion(states, probs, "conclusive")
    1.6824720366950206e-09

    Consider the following two Bell states

    .. math::
        \begin{equation}
            \begin{aligned}
                u_0 &= \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \\
                u_1 &= \frac{1}{\sqrt{2}} \left( |00 \rangle - |11 \rangle \right).
            \end{aligned}
        \end{equation}

    For the corresponding density matrices :math:`\rho_0 = u_0 u_0^*` and
    :math:`\rho_1 = u_1 u_1^*`, we may construct a set

    .. math::
        \rho = \{\rho_0, \rho_1 \}

    such that

    .. math::
        p = \{1/2, 1/2\}.

    It is not possible to unambiguously exclude either of the two states. We can see that the result
    of the function in :code:`toqito` yields a value of :math:`0` as the probability for this to
    occur.

    >>> from toqito.state_opt import state_exclusion
    >>> from toqito.states import bell
    >>> import numpy as np
    >>> rho1 = bell(0) * bell(0).conj().T
    >>> rho2 = bell(1) * bell(1).conj().T
    >>>
    >>> states = [rho1, rho2]
    >>> probs = [1/2, 1/2]
    >>>
    >>> state_exclusion(states, probs, "unambiguous")
    -7.250173600116328e-18

    References
    ==========
    .. [PBR12] "On the reality of the quantum state"
        Pusey, Matthew F., Barrett, Jonathan, and Rudolph, Terry.
        Nature Physics 8.6 (2012): 475-478.
        arXiv:1111.3328

    .. [BJOP14] "Conclusive exclusion of quantum states"
        Bandyopadhyay, Somshubhro, Jain, Rahul, Oppenheim, Jonathan,
        Perry, Christopher
        Physical Review A 89.2 (2014): 022336.
        arXiv:1306.4683

    :param states: A list of states provided as either matrices or vectors.
    :param probs: Respective list of probabilities each state is selected.
    :param method: Exclusion method (either `conclusive` or `unambiguous`.
    :return: The optimal probability with which Bob can guess the state he was
             not given from `states`.
    """
    obj_func = []
    measurements = []
    constraints = []

    __is_states_valid(states)
    if probs is None:
        probs = [1 / len(states)] * len(states)
    __is_probs_valid(probs)

    supported_methods = ["conclusive", "unambiguous"]
    if method not in supported_methods:
        raise ValueError(
            f"Exclusion method {method} not supported. Please "
            f"select one from {supported_methods}."
        )

    dim_x, dim_y = states[0].shape

    # The variable `states` is provided as a list of vectors. Transform them
    # into density matrices.
    if dim_y == 1:
        for i, state_ket in enumerate(states):
            states[i] = state_ket * state_ket.conj().T

    for i, _ in enumerate(states):
        measurements.append(cvxpy.Variable((dim_x, dim_x), PSD=True))

        obj_func.append(probs[i] * cvxpy.trace(states[i].conj().T @ measurements[i]))

        if method == "unambiguous":
            constraints.append(cvxpy.trace(states[i] @ measurements[i]) == 0)

    if method == "conclusive":
        constraints.append(sum(measurements) == np.identity(dim_x))
    elif method == "unambiguous":
        constraints.append(sum(measurements) <= np.identity(dim_x))

    if method == "conclusive":
        if np.iscomplexobj(states[0]):
            objective = cvxpy.Minimize(cvxpy.real(sum(obj_func)))
        else:
            objective = cvxpy.Minimize(sum(obj_func))
    elif method == "unambiguous":
        if np.iscomplexobj(states[0]):
            objective = cvxpy.Maximize(cvxpy.real(sum(obj_func)))
        else:
            objective = cvxpy.Maximize(sum(obj_func))

    problem = cvxpy.Problem(objective, constraints)
    sol_default = problem.solve()

    return 1 / len(states) * sol_default
