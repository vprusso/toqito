"""Determines whether a quantum channel is extremal."""

import numpy as np
from numpy.linalg import matrix_rank

from toqito.channel_ops.choi_to_kraus import choi_to_kraus


def is_extremal(phi: np.ndarray | list[np.ndarray | list[np.ndarray]], tol: float = 1e-9) -> bool:
    """Determine whether a quantum channel is extremal.

    A channel with Kraus operators {A_i} is extremal if and only if
    the set {A_i^† A_j : i,j=1,...,r} is linearly independent.

    The channel can be provided as either:
      - A Choi matrix (np.ndarray), which is converted to a flat list of Kraus operators.
      - A flat list of Kraus operators ([np.ndarray, ...]).
      - A nested list of Kraus operators ([[np.ndarray, ...], ...]), in which case the
        list is flattened.

    :param phi: The quantum channel, either a list (possibly nested) of Kraus operators or a Choi matrix.
    :type phi: list[numpy.ndarray] | list[list[numpy.ndarray]] | numpy.ndarray
    :raises ValueError: If the input is neither a valid list of Kraus operators nor a Choi matrix.
    :return: True if the channel is extremal; False otherwise.
    :rtype: bool
    """
    # If input is a Choi matrix, convert to a (flat) list of Kraus operators.
    if isinstance(phi, np.ndarray):
        kraus_ops = choi_to_kraus(phi)
    elif isinstance(phi, list):
        # If the first element is a list, assume nested list of Kraus operators.
        if len(phi) == 0:
            raise ValueError("The channel must contain at least one Kraus operator.")
        if isinstance(phi[0], list):
            # Flatten the nested list
            kraus_ops = [op for sublist in phi for op in sublist if isinstance(op, np.ndarray)]
        elif all(isinstance(op, np.ndarray) for op in phi):
            kraus_ops = phi
        else:
            raise ValueError("Channel must be a list (or nested list) of Kraus operators.")
    else:
        raise ValueError("Channel must be a list of Kraus operators or a Choi matrix.")

    # Check that we have at least one Kraus operator.
    if not kraus_ops:
        raise ValueError("The channel must contain at least one Kraus operator.")

    r = len(kraus_ops)

    # A single Kraus operator (e.g., a unitary channel) is always extremal.
    if r == 1:
        return True

    # Compute the set {A_i^† A_j} for every pair (i, j).
    flattened_products = [
        np.dot(A.conj().T, B).flatten() for A in kraus_ops for B in kraus_ops
    ]

    # Form a matrix whose columns are these vectorized operators.
    M = np.column_stack(flattened_products)

    # The channel is extremal if and only if the operators {A_i^† A_j} are linearly independent,
    # i.e. the rank of M equals r^2.
    return matrix_rank(M, tol=tol) == r * r
