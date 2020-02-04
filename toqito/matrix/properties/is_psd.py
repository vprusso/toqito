"""Determines whether or not a matrix is positive semidefinite."""
import numpy as np


def is_psd(mat: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check if matrix is positive semidefinite (PSD).

    :param mat: Matrix to check.
    :param tol: Tolerance for numerical accuracy.
    :return: Return True if matrix is PSD, and False otherwise.
    """
    return np.all(np.linalg.eigvalsh(mat) > -tol)
