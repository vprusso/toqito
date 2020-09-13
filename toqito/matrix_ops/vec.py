"""Vec operation."""
import numpy as np


def vec(mat: np.ndarray) -> np.ndarray:
    r"""
    Perform the vec operation on a matrix [WATVEC]_.

    Stacks the rows of the matrix on top of each other to
    obtain the "vec" representation of the matrix.

    The vec function is a linear mapping that in essence converts each row to
    column, and then continually stacks the columns on top of each other.
    An example is helpful.

    For instance, for the following matrix:

    .. math::
        X =
        \begin{pmatrix}
            1 & 2 \\
            3 & 4
        \end{pmatrix}

    it holds that

    .. math::
        \text{vec}(X) = \begin{pmatrix} 1 & 2 & 3 & 4 \end{pmatrix}^T

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
        \text{vec}(A) = \left[1, 2, 3, 4 \right]^{T}.

    >>> from toqito.matrix_ops import vec
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4]])
    >>> vec(X)
    [[1],
     [3],
     [2],
     [4]]

    See Also
    ========
    unvec

    References
    ==========
    .. [WATVEC] Watrous, John.
        "The theory of quantum information."
        Section: "The operator-vector correspondence".
        Cambridge University Press, 2018.

    :param mat: The input matrix.
    :return: The vec representation of the matrix.
    """
    return mat.reshape((-1, 1), order="F")
