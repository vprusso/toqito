"""Kp-norm for matrices."""

import numpy as np


def kp_norm(mat: np.ndarray, k: int, p: int) -> float:
    r"""Compute the kp_norm of vector or matrix.

    Calculate the p-norm of a vector or the k-largest singular values of a
    matrix.

    Examples
    ========
    To compute the p-norm of a vector:

    >>> import numpy as np
    >>> from toqito.matrix_props import kp_norm
    >>> from toqito.states import bell
    >>> '%.2f' % kp_norm(bell(0), 1, np.inf)
    '1.00'


    To compute the k-largest singular values of a matrix:

    >>> import numpy as np
    >>> from toqito.matrix_props import kp_norm
    >>> from toqito.rand import random_unitary
    >>> '%.2f' % kp_norm(random_unitary(5), 5, 2)
    '2.24'

    .. note::
        You do not need to use `'%.2f' %` when you use this function.
        We use this to format our output such that `doctest` compares the calculated output to the
        expected output upto two decimal points only. The accuracy of the solvers can calculate the
        `float` output to a certain amount of precision such that the value deviates after a few digits
        of accuracy.

    :param mat: 2D numpy ndarray
    :param k: The number of singular values to take.
    :param p: The order of the norm.
    :return: The kp-norm of a matrix.

    """
    dim = min(mat.shape)

    # If the requested norm is the Frobenius norm, compute it using numpy's
    # built-in Frobenius norm calculation, which is significantly faster than
    # computing singular values.
    if k >= dim and p == 2:
        return np.linalg.norm(mat, ord="fro")

    s_vals = np.linalg.svd(mat, compute_uv=False)
    return np.linalg.norm(s_vals[:k], ord=p)
