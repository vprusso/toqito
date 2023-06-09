"""Is matrix diagonally dominant."""
import numpy as np
from toqito.matrix_props import is_square


def is_diagonally_dominant(mat: np.ndarray, is_strict=True) -> bool:
    r"""
       Check if matrix is diagnal dominant (DD) [WikPD]_.

       Examples
       ==========

       Consider the following matrix

       .. math::
           A = \begin{pmatrix}
                   2 & -1 & 0 \\
                   0 & 2 & -1 \\
                   0 & -1 & 2
               \end{pmatrix}

       our function indicates that this is indeed a diagonally dominant matrix.

       >>> from toqito.matrix_props import is_diagonally_dominant
       >>> import numpy as np
       >>> A = np.array([[2, -1, 0], [0, 2, -1], [0, -1, 2]])
       >>> is_diagonally_dominant(A)
       True

       Alternatively, the following example matrix :math:`B` defined as

       .. math::
           B = \begin{pmatrix}
                   -1 & 2 \\
                   -1 & -1
               \end{pmatrix}

       is not positive definite.

       >>> from toqito.matrix_props import is_positive_definite
       >>> import numpy as np
       >>> B = np.array([[-1, 2], [-1, -1]])
       >>> is_diagonally_dominant(B)
       False

       References
       ==========
       .. [WikPD] Wikipedia: Definiteness of a matrix.
           https://en.wikipedia.org/wiki/Diagonally_dominant_matrix

       :param mat: Matrix to check.
       :param is_strict: Wether the inequlity is strict.
       :return: Return :code:`True` if matrix is diagnally dominant, and :code:`False` otherwise.
       """
    if not is_square(mat):
        return False

    mat = np.abs(mat)
    diag_coeffs = np.diag(mat)
    row_sum = np.sum(mat, axis=1) - diag_coeffs
    return np.all(diag_coeffs > row_sum) if is_strict else np.all(diag_coeffs >= row_sum)