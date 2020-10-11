"""Compute a list of Kraus operators from the Choi matrix."""
from typing import List
import numpy as np

from toqito.matrix_ops import unvec


def choi_to_kraus(choi_mat: np.ndarray, tol: float = 1e-9) -> List[List[np.ndarray]]:
    r"""
    Compute a list of Kraus operators from the Choi matrix [Rigetti20]_.

    Note that unlike the Choi or natural representation of operators, the Kraus representation is
    *not* unique.

    This function has been adapted from [Rigetti20]_.

    Examples
    ========

    Consider taking the Kraus operators of the Choi matrix that characterizes the "swap operator"
    defined as

    .. math::
        \begin{pmatrix}
        \end{pmatrix}

    The corresponding Kraus operators of the swap operator are given as follows,

    .. math::
        \begin{equation}
            \begin{aligned}
                \frac{1}{\sqrt{2}}
                \begin{pmatrix}
                    0 & i \\ -i & 0
                \end{pmatrix}, &\quad
                \frac{1}{\sqrt{2}}
                \begin{pmatrix}
                    0 & 1 \\
                    1 & 0
                \end{pmatrix}, \\
                \begin{pmatrix}
                    1 & 0 \\
                    0 & 0
                \end{pmatrix}, &\quad
                \begin{pmatrix}
                    0 & 0 \\
                    0 & 1
                \end{pmatrix}.
            \end{aligned}
        \end{equation}

    This can be verified in :code:`toqito` as follows.

    >>> import numpy as np
    >>> from toqito.channel_ops import choi_to_kraus
    >>> choi_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    >>> kraus_ops = choi_to_kraus(choi_mat)
    >>> kraus_ops
    [array([[ 0.+0.j        ,  0.+0.70710678j],
           [-0.-0.70710678j,  0.+0.j        ]]), array([[0.        , 0.70710678],
           [0.70710678, 0.        ]]), array([[1., 0.],
           [0., 0.]]), array([[0., 0.],
           [0., 1.]])]

    See Also
    ========
    kraus_to_choi

    References
    ==========
    .. [Rigetti20] Forest Benchmarking (Rigetti).
        https://github.com/rigetti/forest-benchmarking

    :param choi_mat: a dim**2 by dim**2 choi matrix
    :param tol: optional threshold parameter for eigenvalues/kraus ops to be discarded
    :return: List of Kraus operators
    """
    eigvals, v_mat = np.linalg.eigh(choi_mat)
    return [
        np.lib.scimath.sqrt(eigval) * unvec(np.array([evec]).T)
        for eigval, evec in zip(eigvals, v_mat.T)
        if abs(eigval) > tol
    ]
