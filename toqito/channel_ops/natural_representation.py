"""Kraus operators to natural representation."""

import numpy as np

from toqito.matrix_ops import tensor


def natural_representation(kraus_ops):
    """Convert a set of Kraus operators to the natural representation of a quantum channel.

    The natural representation of a quantum channel is given by:
    Φ = ∑_i K_i ⊗ K_i*
    where K_i* is the complex conjugate of K_i.

    :args: kraus_ops (list[np.ndarray]): List of Kraus operators.
    :return: np.ndarray: The natural representation of the quantum channel.

    Examples:
    >>> import numpy as np
    >>> # Kraus operators for a depolarizing channel
    >>> k0 = np.sqrt(1/2) * np.array([[1, 0], [0, 1]])
    >>> k1 = np.sqrt(1/2) * np.array([[0, 1], [1, 0]])
    >>> nat_rep = natural_representation([k0, k1])
    >>> print(nat_rep)
    >>> print(nat_rep)
    [[0.5 0.  0.  0.5]
     [0.  0.5 0.5 0. ]
     [0.  0.5 0.5 0. ]
     [0.5 0.  0.  0.5]]

    """
    if not isinstance(kraus_ops, list):
        raise ValueError("Kraus operators must be provided as a list.")

    if not all(isinstance(k, np.ndarray) for k in kraus_ops):
        raise ValueError("All Kraus operators must be NumPy arrays.")

    if len(kraus_ops) == 0:
        raise ValueError("At least one Kraus operator must be provided.")

    dim = kraus_ops[0].shape
    if not all(k.shape == dim for k in kraus_ops):
        raise ValueError("All Kraus operators must have the same dimensions.")

    # Compute the natural representation
    return np.sum([tensor(k, np.conjugate(k)) for k in kraus_ops], axis=0)
