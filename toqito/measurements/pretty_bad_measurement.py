"""Compute the set of pretty bad measurements from an ensemble."""
from typing import Optional

import numpy as np
from toqito.measurements import pretty_good_measurement


def pretty_bad_measurement(states: list[np.ndarray], probs: Optional[list[float]] = None) -> list[np.ndarray]:
    r"""Return the set of pretty bad measurements from a set of vectors and corresponding probabilities.

    This computes the "pretty bad measurement" is defined in :cite:`Hughston_1993_Complete,` and is an analogous idea to
    the "pretty good measurement" from :cite:`McIrvin_2024_Pretty,`. The "pretty bad measurement" is useful in the
    context of state exclusion where the pretty good measurmennt is often used for minimum-error quantum state
    discrimination.

    See Also
    ========
    pretty_good_measurement

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
    [array([[0.16666667, 0.        ],
        [0.        , 0.5       ]]), array([[ 0.41666667, -0.14433757],
        [-0.14433757,  0.25      ]]), array([[0.41666667, 0.14433757],
        [0.14433757, 0.25      ]])]

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

    pgm = pretty_good_measurement(states, probs)
    dim = pgm[0].shape[0]

    return [1 / (n - 1) * (np.identity(dim) - pgm[i]) for i in range(n)]

