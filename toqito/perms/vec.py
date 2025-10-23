"""Performs the vec operation on a matrix."""

import numpy as np


def vec(mat: np.ndarray) -> np.ndarray:
    r"""Perform the vec operation on a matrix.

    For more info, see Section: The Operator-Vector Correspondence from :footcite:`Watrous_2018_TQI`.

    The function reorders the given matrix into a column vector by stacking the columns of the matrix sequentially.

    For instance, for the following matrix:

    .. math::
        X =
        \begin{pmatrix}
            1 & 2 \\
            3 & 4
        \end{pmatrix}

    it holds that

    .. math::
        \text{vec}(X) = \begin{pmatrix} 1 & 3 & 2 & 4 \end{pmatrix}^T

    More formally, the vec operation is defined by

    .. math::
        \text{vec}(E_{a,b}) = e_a \otimes e_b

    for all :math:`a` and :math:`b` where

    .. math::
        E_{a,b}(c,d) = \begin{cases}
                          1 & \text{if} \ (c,d) = (a,b) \\
                          0 & \text{otherwise}
                        \end{cases}

    for all :math:`c` and :math:`d` and where

    .. math::
        e_a(b) = \begin{cases}
                     1 & \text{if} \ a = b \\
                     0 & \text{if} \ a \not= b
                 \end{cases}

    for all :math:`a` and :math:`b`.

    Examples
    ==========

    Consider the following matrix

    .. math::
        A = \begin{pmatrix}
                1 & 2 \\
                3 & 4
            \end{pmatrix}

    Performing the :math:`\text{vec}` operation on :math:`A` yields

    .. math::
        \text{vec}(A) = \left[1, 3, 2, 4 \right]^{T}.

    .. jupyter-execute::

     import numpy as np
     from toqito.perms import vec

     X = np.array([[1, 2], [3, 4]])

     vec(X)

    See Also
    ========
    :py:func:`~toqito.matrix_ops.unvec.unvec`

    References
    ==========
    .. footbibliography::



    :param mat: The input matrix.
    :return: The vec representation of the matrix.

    """
    return mat.reshape((-1, 1), order="F")
