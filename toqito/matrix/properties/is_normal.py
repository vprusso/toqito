"""Determines whether or not a matrix is normal."""
import numpy as np


def is_normal(mat: np.ndarray,
              rtol: float = 1e-05,
              atol: float = 1e-08) -> bool:
    r"""
    Determine if a matrix is normal.

    A matrix is normal if it commutes with its adjoint

    :math: `[X, X^*] = 0`,

    or, equivalently if

    :math: `X^* X = X X^*`.

    References:
    [1] Wikipedia: Normal matrix.
        https://en.wikipedia.org/wiki/Normal_matrix

    :param mat: The matrix to check.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: Returns True if the matrix is normal and False otherwise.
    """
    return np.allclose(np.matmul(mat, mat.conj().T),
                       np.matmul(mat.conj().T, mat),
                       rtol=rtol,
                       atol=atol)
