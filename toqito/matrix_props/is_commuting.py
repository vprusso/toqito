"""Is matrix commuting."""
import numpy as np


def is_commuting(mat_1: np.ndarray, mat_2: np.ndarray) -> bool:
    r"""
    Determine if two linear operators commute with each other [WikCom]_.

    For any pair of operators :math:`X, Y \in \text{L}(\mathcal{X})`, the
    Lie bracket :math:`\left[X, Y\right] \in \text{L}(\mathcal{X})` is defined
    as

    .. math::
        \left[X, Y\right] = XY - YX.

    It holds that :math:`` if and only if :math:`X` and :math:`Y` commute
    [WatCom18]_.

    Examples
    ==========

    Consider the following matrices:

    .. math::
        A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix},
        \quad \text{and} \quad
        B = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}.

    It holds that :math:`AB=0`, however

    .. math::
        BA = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix} = A,

    and hence, do not commute.

    >>> from toqito.matrix_props import is_commuting
    >>> import numpy as np
    >>> mat_1 = np.array([[0, 1], [0, 0]])
    >>> mat_2 = np.array([[1, 0], [0, 0]])
    >>> is_commuting(mat_1, mat_2)
    False

    Consider the following pair of matrices:

    .. math::
        A = \begin{pmatrix}
            1 & 0 & 0 \\
            0 & 1 & 0 \\
            1 & 0 & 2
            \end{pmatrix}, \quad \text{and} \quad
        B = \begin{pmatrix}
            2 & 4 & 0 \\
            3 & 1 & 0 \\
            -1 & -4 & 1
            \end{pmatrix}.

    It may be verified that :math:`AB = BA = 0`, and therefore :math`A` and
    :math:`B` commute.

    >>> from toqito.matrix_props import is_commuting
    >>> import numpy as np
    >>> mat_1 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 2]])
    >>> mat_2 = np.array([[2, 4, 0], [3, 1, 0], [-1, -4, 1]])
    >>> is_commuting(mat_1, mat_2)
    True

    References
    ==========
    .. [WikCom] Wikipedia: Commuting matrices
        https://en.wikipedia.org/wiki/Commuting_matrices

    .. [WatCom18] Watrous, John.
        "The theory of quantum information."
        Section: "Lie brackets and commutants".
        Cambridge University Press, 2018.

    :param mat_1: First matrix to check.
    :param mat_2: Second matrix to check.
    :return: Return `True` if :code:`mat_1` commutes with :code:`mat_2` and False otherwise.
    """
    return np.allclose(mat_1 @ mat_2 - mat_2 @ mat_1, 0)
