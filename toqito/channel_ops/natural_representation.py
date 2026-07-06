"""Kraus operators to natural representation."""

import numpy as np

from toqito.matrix_ops import tensor


def natural_representation(kraus_ops: list[np.ndarray]) -> np.ndarray:
    r"""Convert a set of Kraus operators to the natural representation of a quantum channel.

    The natural representation of a quantum channel is given by:
    \(\Phi = \sum_i K_i \otimes K_i^*\)
    where \(K_i^*\) is the complex conjugate of \(K_i\).

    Examples:
        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channel_ops import natural_representation
        k0 = np.sqrt(1/2) * np.array([[1, 0], [0, 1]])
        k1 = np.sqrt(1/2) * np.array([[0, 1], [1, 0]])
        print(natural_representation([k0, k1]))
        ```

    """
    if not kraus_ops:
        raise ValueError("The list of Kraus operators cannot be empty.")

    dim = kraus_ops[0].shape
    if not all(k.shape == dim for k in kraus_ops):
        raise ValueError("All Kraus operators must have the same dimensions.")

    # Compute the natural representation.
    natural_rep = tensor(kraus_ops[0], np.conjugate(kraus_ops[0]))
    for k_mat in kraus_ops[1:]:
        natural_rep += tensor(k_mat, np.conjugate(k_mat))
    return natural_rep
