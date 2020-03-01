"""Calculates probability of state exclusion."""
from typing import List
import cvxpy as cvx
import numpy as np


def state_exclusion(states: List[np.ndarray],
                    probs: List[float] = None) -> float:
    r"""
    Compute probability of state exclusion.

    The "quantum state exclusion" problem involves a collection of :math: `n`
    quantum states

    ..math::
        `\rho = \{ \rho_0, \ldots, \rho_n \},`

    as well as a list of corresponding probabilities

    ..math::
        `p = \{ p_0, \ldots, p_n \}`

    Alice chooses :math: `i` with probability `p_i` and creates the state
    :math: `rho_i`

    Bob wants to guess which state he was *not* given from the collection of
    states. State exclusion implies that ability to discard (with certainty) at
    least one out of the "n" possible quantum states by applying a measurement.

    This function implements the following semidefinite program that provides
    the optimal probability with which Bob can conduct quantum state exclusion.

    ..math::
        ````
        \begin{align*}
            \text{minimize:} \quad & \sum_{i=0}^n p_i \ip{M_i}{\rho_i} \\
            \text{subject to:} \quad & M_0 + \ldots + M_n = \mathbb{I},\\
                                     & M_0, \ldots, M_n >= 0
            \end{align*}
        ```

    References:
        [1] "On the reality of the quantum state"
            Pusey, Matthew F., Jonathan Barrett, and Terry Rudolph.
            Nature Physics 8.6 (2012): 475-478.
            arXiv:1111.3328
        [2] "Conclusive exclusion of quantum states"
            Bandyopadhyay, Somshubhro, et al.
            Physical Review A 89.2 (2014): 022336.
            arXiv:1306.4683

    :param states: A list of density operators (matrices) corresponding to
                   quantum states.
    :param probs: A list of probabilities where `probs[i]` corresponds to the
                  probability that `states[i]` is selected by Alice.
    :return: The optimal probability with which Bob can guess the state he was
             not given from `states`.
    """
    # Assume that at least one state is provided.
    if states is None or states == []:
        raise ValueError("InvalidStates: There must be at least one state "
                         "provided.")

    # Assume uniform probability if no specific distribution is given.
    if probs is None:
        probs = [1/len(states)] * len(states)
    if sum(probs) != 1:
        raise ValueError("Invalid: Probabilities must sum to 1.")

    dim = states[0].shape

    obj_func = []
    measurements = []
    constraints = []
    for i, _ in enumerate(states):
        measurements.append(cvx.Variable(dim, PSD=True))

        obj_func.append(probs[i] * cvx.trace(
            states[i].conj().T @ measurements[i]))

    constraints.append(sum(measurements) == np.identity(dim[0]))

    objective = cvx.Minimize(sum(obj_func))
    problem = cvx.Problem(objective, constraints)
    sol_default = problem.solve()

    return 1/len(states) * sol_default
