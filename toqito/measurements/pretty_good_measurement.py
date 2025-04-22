"""Compute the set of pretty good measurements from an ensemble."""

import numpy as np
import scipy

from toqito.matrix_ops import to_density_matrix


def pretty_good_measurement(states: list[np.ndarray], probs: list[float] | None = None, tol: float = 1e-8) -> list[np.ndarray]:
    r"""Return the set of pretty good measurements from a set of vectors and corresponding probabilities.

    This computes the "pretty good measurement" as initially defined in :cite:`Hughston_1993_Complete`.

    The pretty good measurement (PGM) (also known as the "square root measurement") is the set of POVMs :math:`(G_1,
    \ldots, G_n)` such that

    .. math::
        G_i = P^{-1/2} \left(p_i \rho_i\right) P^{-1/2} \quad \text{where} \quad
        P = \sum_{i=1}^n p_i \rho_i.

    See Also
    ========
    :func:`.pretty_bad_measurement`

    Examples
    ========
    Consider the collection of trine states.

    .. math::
        u_0 = |0\rangle, \quad
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
    :param probs: A set of fixed probabilities for a given ensemble of quantum states.
                  The function assumes a uniform probability distribution if the fixed
                  probabilities for the input ensemble are not provided.
    :param tol: A tolerance value for numerical comparisons.
    :return A list of matrices representing the pretty good measurement.

    """
    n = len(states)

    # If not probabilities are explicitly given, assume a uniform distribution.
    if probs is None:
        probs = n * [1 / n]

    if len(states) != len(probs):
        raise ValueError(f"Number of states {len(states)} must be equal to number of probabilities {len(probs)}")

    if not np.isclose(sum(probs), 1):
        raise ValueError("Probability vector should sum to 1.")

    states = [to_density_matrix(state) for state in states]

    # 1. Assemble the average state.
    p_var = sum(probs[i] * states[i] for i in range(n))

    # 2. Diagonalize.
    vals, vecs = np.linalg.eigh(p_var)

    # 3. Invert only the non‑zero eigenvalues.
    inv_sqrt_vals = np.array([1/np.sqrt(v) if v > tol else 0.0 for v in vals])

    # 4. Reconstruct P^{-1/2}.
    P_inv_sqrt = vecs @ np.diag(inv_sqrt_vals) @ vecs.conj().T

    # 5. Build PGM measurements.
    return [
        P_inv_sqrt @ (probs[i] * states[i]) @ P_inv_sqrt
        for i in range(n)
    ]
