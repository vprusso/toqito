"""Outer product operation."""
import numpy as np


def outer_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    r"""Compute the outer product :math:`|v_1\rangle\langle v_2|` of two vectors.

    The outer product is calculated as follows :cite:`WikiOuterProd` :

    .. math::
        \left|\begin{pmatrix}a_1\\\vdots\\a_n\end{pmatrix}\right\rangle\left\langle\begin{pmatrix}b_1\\\vdots\\b_n\end{pmatrix}\right|=\begin{pmatrix}a_1\\\vdots\\a_n\end{pmatrix}\begin{pmatrix}b_1&\cdots&b_n\end{pmatrix}=\begin{pmatrix}a_1b_1&\cdots&a_1b_n\\\vdots&\ddots&\vdots\\a_1b_n&\cdots&a_nb_n\end{pmatrix}

    Example
    ==========
    The outer product of the vectors :math:`v1 = \begin{pmatrix}1 \\ 2 \\ 3 \end{pmatrix}` and :math:`v2 =
    \begin{pmatrix}4 \\ 5 \\ 6 \ \end{pmatrix}` looks as follows:

    .. math::
        \left|\begin{pmatrix}
            1 \\ 2 \\ 3
        \end{pmatrix}\right\rangle
        \left\langle
        \begin{pmatrix}
            4 \\ 5 \\ 6
        \end{pmatrix}\right|=
        \begin{pmatrix}
            1 \\ 2 \\ 3
        \end{pmatrix}
        \begin{pmatrix}
            4 & 5 & 6
        \end{pmatrix}=
        \begin{pmatrix}
            1 \times 4 & 1 \times 5 & 1 \times 6 \\
            2 \times 4 & 2 \times 5 & 2 \times 6 \\
            3 \times 4 & 3 \times 5 & 3 \times 6
        \end{pmatrix}=
        \begin{pmatrix}
            4 & 5 & 6 \\
            8 & 10 & 12 \\
            12 & 15 & 18
        \end{pmatrix}

    In :code:`toqito`, this looks like this:

    >>> import numpy as np
    >>> from toqito.matrix_ops import outer_product
    >>> v1, v2 = np.array([1,2,3]), np.array([4,5,6])
    >>> outer_product(v1,v2)
    [[4, 5, 6],
     [8, 10, 12],
     [12, 15, 18]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: Vector dimensions are mismatched.
    :param v1: v1 and v2, both vectors of dimensions :math:`(n,1)` where :math:`n>1`.
    :param v2: v1 and v2, both vectors of dimensions :math:`(n,1)` where :math:`n>1`.
    :return: The computed outer product.

    """
    if v1.ndim != 1 or v2.ndim != 1:
        raise ValueError("Both v1 and v2 must be 1D vectors.")
    if v1.shape[0] != v2.shape[0]:
        raise ValueError("Dimension mismatch between v1 and v2.")

    return np.outer(v1, np.conj(v2))
