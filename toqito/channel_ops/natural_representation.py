"""Kraus operators to natural representation."""

import numpy as np

from toqito.matrix_ops import tensor


def natural_representation(kraus_ops: list[np.ndarray]) -> np.ndarray:
    r"""Convert a set of Kraus operators to the natural representation of a quantum channel.

    The natural representation of a quantum channel is given by:
    :math:`\Phi = \sum_i K_i \otimes K_i^*`
    where :math:`K_i^*` is the complex conjugate of :math:`K_i`.

    Examples
    ==========
    .. jupyter-execute::

     import numpy as np
     from toqito.channel_ops import natural_representation
     k0 = np.sqrt(1/2) * np.array([[1, 0], [0, 1]])
     k1 = np.sqrt(1/2) * np.array([[0, 1], [1, 0]])
     print(natural_representation([k0, k1]))


    """
    dim = kraus_ops[0].shape
    if not all(k.shape == dim for k in kraus_ops):
        raise ValueError("All Kraus operators must have the same dimensions.")

    # Compute the natural representation.
    return np.sum([tensor(k, np.conjugate(k)) for k in kraus_ops], axis=0)
