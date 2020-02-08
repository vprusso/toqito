"""Converts a state vector or density matrix representation of a state to a
density matrix."""
import numpy as np


def pure_to_mixed(phi: np.ndarray) -> np.ndarray:
    """
    Converts a state vector or density matrix representation of a state to a
    density matrix.

    :param phi: A density matrix or a pure state vector.
    :return: density matrix representation of `phi`, regardless of
             whether `phi` is itself already a density matrix or if
             if is a pure state vector.
    """

    # Compute the size of `phi`. If it's already a mixed state, leave it alone.
    # If it's a vector (pure state), make it into a density matrix.
    row_dim, col_dim = phi.shape[0], phi.shape[1]

    # It's a pure state vector.
    if min(row_dim, col_dim) == 1:
        return phi * phi.conj().T
    # It's a density matrix.
    if row_dim == col_dim:
        return phi
    # It's neither.
    msg = """
        InvalidDim: `phi` must be either a vector or square matrix.
    """
    raise ValueError(msg)
