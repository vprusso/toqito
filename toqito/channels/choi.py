"""Generates the Choi channel."""

import numpy as np

from toqito.states import max_entangled


def choi(a_var: int = 1, b_var: int = 1, c_var: int = 0) -> np.ndarray:
    r"""Produce the Choi channel or one of its generalizations :cite:`Choi_1992_Generalized`.

    The *Choi channel* is a positive map on 3-by-3 matrices that is capable of detecting some
    entanglement that the transpose map is not.

    The standard Choi channel defined with :code:`a=1`, :code:`b=1`, and :code:`c=0` is the Choi
    matrix of the positive map defined in :cite:`Choi_1992_Generalized`. Many of these maps are capable of detecting
    PPT entanglement.

    Examples
    ==========

    The standard Choi channel is given as

    .. math::
        \Phi_{1, 1, 0} =
        \begin{pmatrix}
            1 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & -1 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            -1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & -1 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            -1 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & 1
        \end{pmatrix}

    We can generate the Choi channel in :code:`toqito` as follows.

    >>> from toqito.channels import choi
    >>> import numpy as np
    >>> choi()
    array([[ 1.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
           [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
           [-1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  1.]])

    The reduction channel is the map :math:`R` defined by:

    .. math::
        R(X) = \text{Tr}(X) \mathbb{I} - X.

    The matrix correspond to this is given as

    .. math::
        \Phi_{0, 1, 1} =
        \begin{pmatrix}
            0 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & -1 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            -1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -1 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            -1 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & 0
        \end{pmatrix}

    The reduction channel is the Choi channel that arises when :code:`a = 0` and when :code:`b =
    c = 1`. We can obtain this matrix using :code:`toqito` as follows.

    >>> from toqito.channels import choi
    >>> import numpy as np
    >>> choi(0, 1, 1)
    array([[ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.],
           [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
           [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
           [-1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.]])

    See Also
    ==========
    reduction

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param a_var: Default integer for standard Choi map.
    :param b_var: Default integer for standard Choi map.
    :param c_var: Default integer for standard Choi map.
    :return: The Choi channel (or one of its  generalizations).

    """
    psi = max_entangled(3, False, False)
    return np.diag([a_var + 1, c_var, b_var, b_var, a_var + 1, c_var, c_var, b_var, a_var + 1]) - psi @ psi.conj().T
