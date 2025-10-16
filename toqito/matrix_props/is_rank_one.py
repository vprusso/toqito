"""Checks if a matrix has rank one."""

import numpy as np


def is_rank_one(mat: np.ndarray, tol: float = 1e-08) -> bool:
    r"""Determine whether the given matrix has rank one :footcite:`WikiMatrixRank`.

    The function evaluates the singular values (equivalently, eigenvalues for Hermitian matrices)
    and counts how many are greater than the provided tolerance.

    Examples
    ========

    Consider the Bell state density matrix :math:`\rho = \ket{\Phi^+}\bra{\Phi^+}`. This matrix
    has rank one.

    .. jupyter-execute::

        from toqito.matrix_props import is_rank_one
        from toqito.states import bell

        rho = bell(0) @ bell(0).conj().T
        is_rank_one(rho)

    On the other hand, the maximally mixed state is not rank one.

    .. jupyter-execute::

        import numpy as np
        from toqito.matrix_props import is_rank_one

        maximally_mixed = np.eye(2) / 2
        is_rank_one(maximally_mixed)

    References
    ==========
    .. footbibliography::

    :param mat: Matrix to test.
    :param tol: Numerical tolerance used when distinguishing non-zero singular values.
    :return: :code:`True` if the matrix has rank at most one, :code:`False` otherwise.

    """
    singular_values = np.linalg.svd(mat, compute_uv=False)
    return np.count_nonzero(singular_values > tol) <= 1
