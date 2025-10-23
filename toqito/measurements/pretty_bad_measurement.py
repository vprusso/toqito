"""Compute the set of pretty bad measurements from an ensemble."""

import numpy as np

from toqito.measurements import pretty_good_measurement


def pretty_bad_measurement(
    states: list[np.ndarray], probs: list[float] | None = None, tol: float = 1e-8
) -> list[np.ndarray]:
    r"""Return the set of pretty bad measurements from a set of vectors and corresponding probabilities.

    This computes the "pretty bad measurement" (PBM) as defined in
    :footcite:`McIrvin_2024_Pretty`. The PBM is an analogue to the "pretty
    good measurement" defined in :footcite:`Belavkin_1975_Optimal,Hughston_1993_Complete`
    and is useful for approximating the optimal measurement
    for state exclusion.

    The PBM is defined in terms of the pretty good measurement (PGM).
    Given the PGM operators :math:`(G_1, \ldots, G_n)`, the corresponding PBM
    is the set of POVMs :math:`(B_1, \ldots, B_n)` where

    .. math::
        B_i = \frac{1}{n - 1} \left(\mathbb{I} - G_i\right).

    See Also
    ========
    :py:func:`~toqito.measurements.pretty_good_measurement.pretty_good_measurement`

    Examples
    ========
    Consider the collection of trine states.

    .. math::
        u_0 = |0\rangle, \quad
        u_1 = -\frac{1}{2}\left(|0\rangle + \sqrt{3}|1\rangle\right), \quad \text{and} \quad
        u_2 = -\frac{1}{2}\left(|0\rangle - \sqrt{3}|1\rangle\right).

    .. jupyter-execute::

     from toqito.states import trine
     from toqito.measurements import pretty_bad_measurement

     states = trine()
     probs = [1 / 3, 1 / 3, 1 / 3]
     pbm = pretty_bad_measurement(states, probs)
     pbm

    References
    ==========
    .. footbibliography::

    :raises ValueError: If number of states does not match number of probabilities.
    :raises ValueError: If probabilities do not sum to 1.
    :param states: A collection of states provided as either vectors or density matrices.
    :param probs: A set of fixed probabilities for each quantum state.
                  If not provided, a uniform distribution is assumed.
    :param tol: A tolerance value for numerical comparisons.
    :return: A list of POVM operators for the PBM.

    """
    n = len(states)

    # If not probabilities are explicitly given, assume a uniform distribution.
    if probs is None:
        probs = n * [1 / n]

    if len(states) != len(probs):
        raise ValueError(f"Number of states {len(states)} must be equal to number of probabilities {len(probs)}")

    if not np.isclose(sum(probs), 1):
        raise ValueError("Probability vector should sum to 1.")

    pbm = pretty_good_measurement(states, probs, tol=tol)
    dim = pbm[0].shape[0]

    return [1 / (n - 1) * (np.identity(dim) - pbm[i]) for i in range(n)]
