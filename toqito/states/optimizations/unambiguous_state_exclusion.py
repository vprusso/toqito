"""Calculates probability of unambiguous state exclusion."""
from typing import List
import cvxpy
import numpy as np


def unambiguous_state_exclusion(
    states: List[np.ndarray], probs: List[float] = None
) -> float:
    r"""
    Compute probability of unambiguous state exclusion [4]_.

    This function implements the following semidefinite program that provides
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
        u_0 = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right) \\
        u_1 = \frac{1}{\sqrt{2}} \left( |00 \rangle - |11 \rangle \right).

    For the corresponding density matrices :math:`\rho_0 = u_0 u_0^*` and
    :math:`\rho_1 = u_1 u_1^*`, we may construct a set

    .. math::
        \rho = \{\rho_0, \rho_1 \}

    such that

    .. math::
        p = \{1/2, 1/2\}.

    It is not possible to unambiguously exclude either of the two states. We can
    see that the result of the function in `toqito` yields a value of :math:`0`
    as the probability for this to occur.

    >>> from toqito.states.optimizations.unambiguous_state_exclusion import
    >>>     unambiguous_state_exclusion
    >>> from toqito.states.states.bell import bell
    >>> import numpy as np
    >>> rho1 = bell(0) * bell(0).conj().T
    >>> rho2 = bell(1) * bell(1).conj().T
    >>>
    >>> states = [rho1, rho2]
    >>> probs = [1/2, 1/2]
    >>>
    >>> unambiguous_state_exclusion(states, probs)
    -7.250173600116328e-18

    References
    ==========
    .. [4] "Conclusive exclusion of quantum states"
        Bandyopadhyay, Somshubhro, et al.
        Physical Review A 89.2 (2014): 022336.
        arXiv:1306.4683

    :param states: A list of density operators (matrices) corresponding to
                   quantum states.
    :param probs: A list of probabilities where `probs[i]` corresponds to the
                  probability that `states[i]` is selected by Alice.
    :return: The optimal probability with which Bob can guess the state he was
             not given from `states` with certainty.
    """
    # Assume that at least one state is provided.
    if states is None or states == []:
        raise ValueError("InvalidStates: There must be at least one state provided.")

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

    obj_func = []
    measurements = []
    constraints = []
    for i, _ in enumerate(states):
        measurements.append(cvxpy.Variable((dim_x, dim_x), PSD=True))

        obj_func.append(probs[i] * cvxpy.trace(states[i].conj().T @ measurements[i]))

        constraints.append(cvxpy.trace(states[i] @ measurements[i]) == 0)

    constraints.append(sum(measurements) <= np.identity(dim_x))

    if np.iscomplexobj(states[0]):
        objective = cvxpy.Maximize(cvxpy.real(sum(obj_func)))
    else:
        objective = cvxpy.Maximize(sum(obj_func))

    problem = cvxpy.Problem(objective, constraints)
    sol_default = problem.solve()

    return 1 / len(states) * sol_default
