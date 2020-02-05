import numpy as np


def is_symmetric(mat: np.ndarray,
                 rtol: float = 1e-05,
                 atol: float = 1e-08) -> bool:
    r"""
    Determines if a matrix is symmetric.

    The following 3x3 matrix is an example of a symmetric matrix:
        :math: `\begin{matrix}
                    1 & 7 & 3 \\
                    7 & 4 & -5 \\
                    3 &-5 & 6
                \end{pmatrix}`

    :param mat: The matrix to check.
    :param rtol: The relative tolerance parameter (default 1e-05).
    :param atol: The absolute tolerance parameter (default 1e-08).
    :return: Returns True if the matrix is symmetric and False otherwise.
    """
    return np.allclose(mat, mat.T, rtol=rtol, atol=atol)
