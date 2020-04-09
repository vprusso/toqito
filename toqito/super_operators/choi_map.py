"""Produces the Choi map or one of its generalizations."""
import numpy as np
from toqito.state.states.max_entangled import max_entangled


def choi_map(a_var: int = 1, b_var: int = 1, c_var: int = 0) -> np.ndarray:
    r"""
    Produce the Choi map or one of its generalizations [1]_.

    The Choi map is a positive map on 3-by-3 matrices that is capable
    of detecting some entanglement that the transpose map is not.

    The standard Choi map defined with `a=1`, `b=1`, and `c=0` is the
    Choi matrix of the positive map defined in [1]. Many of these
    maps are capable of detecting PPT entanglement.

    Examples
    ==========

    The standard Choi map is given as

    .. math::
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

    We can generate the Choi map in `toqito` as follows.

    >>> from toqito.super_operators.choi_map import choi_map
    >>> import numpy as np
    >>> choi_map()
    array([[ 1.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
           [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
           [-1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  1.]])

    The reduction map is the map :math:`R` defined by:

    .. math::
        R(X) = \text{Tr}(X) \mathbb{I} - X.

    The matrix correspond to this is given as

    .. math::
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

    The reduction map is the Choi map that arises when :math:`a = 0`,
    :math:`b = c = 1`. We can obtain this matrix using `toqito` as follows.

    >>> from toqito.super_operators.choi_map import choi_map
    >>> import numpy as np
    >>> choi_map(0, 1, 1)
    array([[ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.],
           [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
           [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
           [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
           [-1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.]])

    References
    ==========
    .. [1] S. J. Cho, S.-H. Kye, and S. G. Lee,
        Linear Alebr. Appl. 171, 213
        (1992).

    :param a_var: Default integer for standard Choi map.
    :param b_var: Default integer for standard Choi map.
    :param c_var: Default integer for standard Choi map.
    """
    psi = max_entangled(3, False, False)
    return (
        np.diag(
            [a_var + 1, c_var, b_var, b_var, a_var + 1, c_var, c_var, b_var, a_var + 1]
        )
        - psi * psi.conj().T
    )
