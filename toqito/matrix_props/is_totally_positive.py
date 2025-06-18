"""Checks if the matrix is totally positive."""

from itertools import combinations

import numpy as np


def is_totally_positive(mat: np.ndarray, tol: float = 1e-6, sub_sizes: list | None = None):
    r"""Determine whether a matrix is totally positive. :footcite:`WikiTotPosMat`.

    A totally positive matrix is a square matrix where all the minors are positive. Equivalently, the determinant of
    every square submatrix is a positive number.

    Examples
    ========
    Consider the matrix

    .. math::
        X = \begin{pmatrix}
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

    Our function indicates that this matrix is indeed totally positive.

    .. jupyter-execute::

        import numpy as np
        from toqito.matrix_props import is_totally_positive

        A = np.array([[1, 2], [3, 4]])

        is_totally_positive(A)

    However, the following example matrix :math:`B` defined as

    .. math::
        B = \begin{pmatrix}
                1 & 2 \\
                3 & -4
            \end{pmatrix}

    is not totally positive. The 2x2 minor of :math:`B` is the determinant of the entire matrix :math:`B`. The
    determinant of :math:`B` is calculated as:

    .. math::
        \text{det}(B) = 1 \times -4 - 2 \times 3 = -4 - 6 = -10

    Since the determinant is negative, :math:`B` is not totally positive.

    .. jupyter-execute::

        import numpy as np
        from toqito.matrix_props import is_totally_positive

        B = np.array([[1, 2], [3, -4]])

        is_totally_positive(B)

    References
    ==========
    .. footbibliography::



    :param mat: Matrix to check.
    :param tol: The absolute tolerance parameter (default 1e-06).
    :param sub_sizes: List of sizes of submatrices to consider. Default is all sizes up to :code:`min(mat.shape)`.
    :return: Return :code:`True` if matrix is totally positive, and :code:`False` otherwise.

    """
    if mat.size == 0:
        raise ValueError("Empty matrix to be neither totally positive nor not totally positive.")

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
                    if d < tol or abs(np.imag(d)) > tol:
                        return False
    return True
