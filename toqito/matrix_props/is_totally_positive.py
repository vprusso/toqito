"""Is matrix totally positive."""
import numpy as np
from itertools import combinations


def is_totally_positive(mat, tol=1e-6, sub_sizes=None):
    """
    Determines whether a matrix is totally positive.

    A totally positive matrix is a square matrix where all the minors are positive. Equivalently, the determinant of
    every square submatrix is a positive number.

    Examples
    ========

    Consider the matrix

    .. math::
        \begin{pmatrix}
            1 & 2 \\
            3 & 4
        \end{pmatrix}
    To determine if this matrix is totally positive, we need to check the positivity of all of its minors. The 1x1
    minors are simply the individual entries of the matrix. For :math:`X`, these are

    .. math::
        \begin{equation}
            \begin{aligned}
                X_{1,1} &= 1 \\
                X_{1,2} &= 2 \\
                X_{2,1} &= 3 \\
                X_{2,2} &= 4 \\
            \end{aligned}
        \end{equation}

    Each of these entries is positive. There is only one 2x2 minor in this case, which is the determinant of the entire
    matrix :math:`X`. The determinant of :math:`X` is calculated as:

    .. math::
        \text{det}(X) = 1 \times 4 - 2 \times 3 = 4 - 6 = 2

    References
    ==========
    .. [WikTotallyPositive] Wikipedia: Totally positive matrix.
        https://en.wikipedia.org/wiki/Totally_positive_matrix

    :param mat: Matrix to check.
    :param atol: The absolute tolerance parameter (default 1e-06).
    :param sub_sizes: List of sizes of submatrices to consider. Default is all sizes up to :code:`min(mat.shape)`.
    :return: Return :code:`True` if matrix is totally positive, and :code:`False` otherwise.
    """
    dims = mat.shape

    if sub_sizes is None:
        sub_sizes = range(1, min(dims) + 1)

    for j in sub_sizes:
        # Handle 1x1 determinants separately.
        if j == 1:
            r, _ = np.where(np.minimum(np.real(mat), -np.abs(np.imag(mat))) < -tol)
            if r.size > 0:
                return False

        # Handle larger determinants.
        else:
            sub_ind_r = list(combinations(range(dims[0]), j))
            sub_ind_c = list(combinations(range(dims[1]), j)) if dims[0] != dims[1] else sub_ind_r

            for kr in sub_ind_r:
                for kc in sub_ind_c:
                    d = np.linalg.det(mat[np.ix_(kr, kc)])
                    if d < -tol or abs(np.imag(d)) > tol:
                        return False
    return True
