"""Kraus operators to natural representation."""

import numpy as np


def natural_representation(kraus_ops: list[np.ndarray]) -> np.ndarray:
    r"""Convert a set of Kraus operators to the natural representation of a quantum channel.

    The natural representation of a quantum channel is given by:
    \(\Phi = \sum_i K_i \otimes K_i^*\)
    where \(K_i^*\) is the complex conjugate of \(K_i\).

    Args:
        kraus_ops: A list of Kraus operators, or a stacked ndarray whose first
            axis indexes the Kraus operators.

    Returns:
        The natural-representation matrix of the quantum channel.

    Examples:
        ```python exec="1" source="above" result="text"
        import numpy as np
        from toqito.channel_ops import natural_representation
        k0 = np.sqrt(1/2) * np.array([[1, 0], [0, 1]])
        k1 = np.sqrt(1/2) * np.array([[0, 1], [1, 0]])
        print(natural_representation([k0, k1]))
        ```

    """
    if len(kraus_ops) == 0:
        raise ValueError("The list of Kraus operators cannot be empty.")

    dim = kraus_ops[0].shape
    if not all(k.shape == dim for k in kraus_ops):
        raise ValueError("All Kraus operators must have the same dimensions.")

    # Compute the natural representation as a single contraction. Stacking the
    # Kraus operators into a (r, out, in) array and contracting
    # \(\sum_i K_i \otimes K_i^*\) with one einsum avoids materializing the r
    # separate Kronecker products before summing.
    stacked = np.asarray(kraus_ops)
    out_dim, in_dim = dim
    return np.einsum("rab,rcd->acbd", stacked, stacked.conj()).reshape(out_dim * out_dim, in_dim * in_dim)
