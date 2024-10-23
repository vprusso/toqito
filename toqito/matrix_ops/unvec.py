"""Unvec operation is used to perform the unvec operation on a vector."""

import numpy as np


def unvec(vector: np.ndarray, shape: list[int] = None) -> np.ndarray:
    r"""Perform the unvec operation on a vector to obtain a matrix :cite:`Rigetti_2022_Forest`.

    Takes a column vector and transforms it into a :code:`shape[0]`-by-:code:`shape[1]` matrix.
    This operation is the inverse of :code:`vec` operation in :code:`toqito`.

    For instance, for the following column vector

    .. math::
        u = \begin{pmatrix} 1 \\ 3 \\ 2 \\ 4 \end{pmatrix},

    it holds that

    .. math::
        \text{unvec}(u) =
        \begin{pmatrix}
            1 & 2 \\
            3 & 4
        \end{pmatrix}

    More formally, the vec operation is defined by

    .. math::
        \text{unvec}(e_a \otimes e_b) = E_{a,b}

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

    This function has been adapted from :cite:`Rigetti_2022_Forest`.

    Examples
    ==========

    Consider the following vector

    .. math::
        u = \begin{pmatrix} 1 \\ 3 \\ 2 \\ 4 \end{pmatrix}

    Performing the :math:`\text{unvec}` operation on :math:`u` yields

    .. math::
        \text{unvec}(u) = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}

    >>> from toqito.matrix_ops import unvec
    >>> import numpy as np
    >>> u = np.array([1, 2, 3, 4])
    >>> unvec(u)
    array([[1, 3],
           [2, 4]])

    See Also
    ========
    vec

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param vector: A (:code:`shape[0] * shape[1]`)-by-1 numpy array.
    :param shape: The shape of the output matrix; by default, the matrix is assumed to be square.
    :return: Returns a :code:`shape[0]`-by-:code:`shape[1]` matrix.

    """
    vector = np.asarray(vector)
    if shape is None:
        dim = int(np.sqrt(vector.size))
        shape = dim, dim
    mat = vector.reshape(*shape, order="F")
    return mat
