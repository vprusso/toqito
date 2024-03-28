"""Compute the set of pretty good measurements from an ensemble."""
from typing import Optional

import numpy as np
import scipy

from toqito.matrix_ops import vector_to_density_matrix


def pretty_good_measurement(states: list[np.ndarray], probs: Optional[list[float]] = None) -> list[np.ndarray]:
    r"""Return the set of pretty good measurements from a set of vectors and corresponding probabilities.

    This computes the "pretty good measurement" as initially defined in :cite:`Hughston_1993_Complete`.

    See Also
    ========
    pretty_bad_measurement

    Examples
    ========
    Consider the collection of trine states.

    .. math::
        u_1 = |0\rangle, \quad
        u_1 = -\frac{1}{2}\left(|0\rangle + \sqrt{3}|1\rangle\right), \quad \text{and} \quad
        u_2 = -\frac{1}{2}\left(|0\rangle - \sqrt{3}|1\rangle\right).

    >>> from toqito.states import trine
    >>> from toqito.measurements import pretty_good_measurement
    >>>
    >>> states = trine()
    >>> probs = [1 / 3, 1 / 3, 1 / 3]
    >>> pgm = pretty_good_measurement(states, probs)
    >>> pgm
    [array([[0.66666667, 0.        ],
           [0.        , 0.        ]]), array([[0.16666667, 0.28867513],
           [0.28867513, 0.5       ]]), array([[ 0.16666667, -0.28867513],
           [-0.28867513,  0.5       ]])]

    References
    ==========
        .. bibliography::
            :filter: docname in docnames


    :raises ValueError: If number of vectors does not match number of probabilities.
    :raises ValueError: If probabilities do not sum to 1.
    :param states: A collection of either states provided as either vectors or density matrices.
    :param probs: A set of probabilities.

    """
    n = len(states)

    # If not probabilities are explicitly given, assume a uniform distribution.
    if probs is None:
        probs = n * [1 / n]

    if len(states) != len(probs):
        raise ValueError(f"Number of states {len(states)} must be equal to number of probabilities {len(probs)}")

    if not np.isclose(sum(probs), 1):
        raise ValueError("Probability vector should sum to 1.")

    states = [vector_to_density_matrix(state) for state in states]
    p_var = sum([probs[i] * states[i] for i in range(n)])

    p_var_sqrt = scipy.linalg.fractional_matrix_power(p_var, -1/2)
    return [p_var_sqrt @ (probs[i] * states[i]) @ p_var_sqrt for i in range(n)]

