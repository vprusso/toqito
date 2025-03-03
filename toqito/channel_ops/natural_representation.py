"""Kraus operators to natural representation."""

from typing import List

import numpy as np

from toqito.matrix_ops import tensor


def natural_representation(kraus_ops: List[np.ndarray]) -> np.ndarray:
    r"""Convert a set of Kraus operators to the natural representation of a quantum channel.

    The natural representation of a quantum channel is given by:
    :math:`\Phi = \sum_i K_i \otimes K_i^*`
    where :math:`K_i^*` is the complex conjugate of :math:`K_i`.

    Examples
    ==========
    >>> import numpy as np
    >>> from toqito.channel_ops import natural_representation
    >>> k0 = np.sqrt(1/2) * np.array([[1, 0], [0, 1]])
    >>> k1 = np.sqrt(1/2) * np.array([[0, 1], [1, 0]])
    >>> print(natural_representation([k0, k1]))
    [[0.5 0.  0.  0.5]
     [0.  0.5 0.5 0. ]
     [0.  0.5 0.5 0. ]
     [0.5 0.  0.  0.5]]

    """
    if not isinstance(kraus_ops, list):
        raise ValueError("Kraus operators must be provided as a list.")

    if not all(isinstance(k, np.ndarray) for k in kraus_ops):
        raise ValuesError("All Kraus operators must be NumPy arrays.")

    if len(kraus_ops) == 0:
        raise ValueError("At least one Kraus operator must be provided.")

    dim = kraus_ops[0].shape
    if not all(k.shape == dim for k in kraus_ops):
        raise ValueError("All Kraus operators must have the same dimensions.")

    # Compute the natural representation
    return np.sum([tensor(k, np.conjugate(k)) for k in kraus_ops], axis=0)
