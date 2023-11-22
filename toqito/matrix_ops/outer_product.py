"""Outer product operation."""
import numpy as np


def outer_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    r"""
    Compute the outer product :math:`|v_1\rangle\langle v_2|` of two vectors.

    The outer product is calculated as follows :cite:`WikiOuterProd` :

    .. math::
        \left|\begin{pmatrix}a_1\\\vdots\\a_n\end{pmatrix}\right\rangle\left\langle\begin{pmatrix}b_1\\\vdots\\b_n\end{pmatrix}\right|=\begin{pmatrix}a_1\\\vdots\\a_n\end{pmatrix}\begin{pmatrix}b_1&\cdots&b_n\end{pmatrix}=\begin{pmatrix}a_1b_1&\cdots&a_1b_n\\\vdots&\ddots&\vdots\\a_1b_n&\cdots&a_nb_n\end{pmatrix}

    Example
    ==========

    The outer product of the vectors :math:`v1 = \begin{pmatrix}1 \\ 2 \\ 3 \end{pmatrix}` and :math:`v2 = \begin{pmatrix}4 \\ 5 \\ 6 \ \end{pmatrix}` looks as follows:

    .. math::
        \left|\begin{pmatrix}1\\2\\3\end{pmatrix}\right\rangle\left\langle\begin{pmatrix}4\\5\\6\end{pmatrix}\right|=\begin{pmatrix}1\\2\\3\end{pmatrix}\begin{pmatrix}4&5&6\end{pmatrix}=\begin{pmatrix}1\times4&1\times5&1\times6\\2\times4&2\times5&2\times6\\3\times4&3\times5&3\times6\end{pmatrix}=\begin{pmatrix}4&5&6\\8&10&12\\12&15&18\end{pmatrix}

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
    :param v1: v1 and v2, both vectors of dimenstions :math:`(n,1)` where :math:`n>1`.
    :param v2: v1 and v2, both vectors of dimenstions :math:`(n,1)` where :math:`n>1`.
    :return: The computed outer product.
    """
    # Check for dimensional validity
    if not (v1.shape[0] == v2.shape[0] and v1.shape[0] > 1 and len(v1.shape) == 1):
        raise ValueError("Dimension mismatch")

    res = np.ndarray((v1.shape[0], v1.shape[0]))
    for i in range(v1.shape[0]):
        for j in range(v1.shape[0]):
            res[i, j] = v1[i] * v2[j]
    return res
