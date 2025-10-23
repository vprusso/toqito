"""Compute an orthonormal basis for the null space of a matrix."""

import numpy as np


def null_space(mat: np.ndarray, tol: float = 1e-08) -> np.ndarray:
    r"""Return an orthonormal basis for the kernel of ``mat`` :footcite:`WikiNullSpace`.

    The routine employs the singular value decomposition so that the columns of the
    returned matrix span the null space and are orthonormal with respect to the
    standard inner product.

    Examples
    ========

    Consider the matrix

    .. math::

        A = \begin{pmatrix} 1 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}.

    Its null space is spanned by the vectors :math:`(1,-1,0)` and :math:`(0,0,1)`.

    .. jupyter-execute::

        import numpy as np
        from toqito.matrix_ops import null_space

        A = np.array([[1, 1, 0], [0, 0, 0]], dtype=float)
        null_basis = null_space(A)
        null_basis

    :param mat: Matrix whose null space is sought.
    :param tol: Numerical tolerance that distinguishes zero singular values.
    :return: A matrix whose columns form an orthonormal basis for the null space.

    """
    mat = np.asarray(mat, dtype=np.complex128)
    if mat.ndim != 2:
        raise ValueError("Input must be a two-dimensional array.")
    if mat.size == 0:
        return np.zeros((mat.shape[0], 0), dtype=np.complex128)

    _, singular_values, vh = np.linalg.svd(mat, full_matrices=True)
    rank = np.sum(singular_values > tol)
    kernel = vh[rank:].conj().T
    if kernel.size == 0:
        return np.zeros((mat.shape[1], 0), dtype=np.complex128)
    q, _ = np.linalg.qr(kernel)
    return q
