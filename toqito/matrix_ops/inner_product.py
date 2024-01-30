"""Inner product operation."""
import numpy as np


def inner_product(v1: np.ndarray, v2: np.ndarray) -> float:
    r"""Compute the inner product :math:`\langle v_1|v_2\rangle` of two vectors :cite:`WikiInnerProd`.

    The inner product is calculated as follows:

    .. math::
        \left\langle
        \begin{pmatrix}
            a_1 \\ \vdots \\ a_n
        \end{pmatrix},
        \begin{pmatrix}
            b_1 \\ \vdots \\ b_n
        \end{pmatrix} \right\rangle =
        \begin{pmatrix}
            a_1, \cdots, a_n
        \end{pmatrix}
        \begin{pmatrix}
            b_1 \\ \vdots \\ b_n
        \end{pmatrix} = a_1 b_1 + \cdots + a_n b_n

    Example
    ==========

    The inner product of the vectors :math:`v1 = \begin{pmatrix}1 \\ 2 \\ 3 \end{pmatrix}` and :math:`v2 =
    \begin{pmatrix}4 \\ 5 \\ 6 \ \end{pmatrix}` looks as follows:

    .. math::
        \left\langle
        \begin{pmatrix}
            1 \\ 2 \\ 3
        \end{pmatrix},
        \begin{pmatrix}
            4 \\ 5 \\ 6
        \end{pmatrix}\right\rangle =
        \begin{pmatrix}
            1, 2, 3
        \end{pmatrix}
        \begin{pmatrix}
            4 \\ 5 \\ 6
        \end{pmatrix} = 1\times 4 + 2\times 5 + 3\times 6 =
        32

    In :code:`toqito`, this looks like this:

    >>> import numpy as np
    >>> from toqito.matrix_ops import inner_product
    >>> v1, v2 = np.array([1,2,3]), np.array([4,5,6])
    >>> inner_product(v1,v2)
    32

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: Vector dimensions are mismatched.
    :param v1: v1 and v2, both vectors of dimensions :math:`(n,1)` where :math:`n>1`.
    :param v2: v1 and v2, both vectors of dimensions :math:`(n,1)` where :math:`n>1`.
    :return: The computed inner product.

    """
    if v1.ndim != 1 or v2.ndim != 1:
        raise ValueError("Both v1 and v2 must be 1D vectors.")
    if v1.shape[0] != v2.shape[0]:
        raise ValueError("Dimension mismatch between v1 and v2.")
    return np.vdot(v1, v2)
