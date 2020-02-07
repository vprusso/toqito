"""Determines whether or not a matrix is a density matrix."""
import numpy as np

from toqito.matrix.properties.is_psd import is_psd


def is_density(mat: np.ndarray) -> bool:
    """
    Check if matrix is a density matrix.

    A matrix is a density matrix if its trace is equal to one and it has the
    property of being positive semidefinite (PSD).

    :param mat: Matrix to check.
    :return: Return `True` if matrix is a density matrix, and `False`
             otherwise.
    """
    return is_psd(mat) and np.isclose(np.trace(mat), 1)
