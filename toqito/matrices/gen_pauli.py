"""Generalized Pauli matrices."""
import numpy as np

from toqito.matrices import shift
from toqito.matrices import clock


def gen_pauli(k_1: int, k_2: int, dim: int) -> np.ndarray:
    r"""
    Produce generalized Pauli operator [WikGenPaul]_.

    Generates a `dim`-by-`dim` unitary operator. More specifically, it is the
    operator `X^k_1*Z^k_2`, where X and Z are the "shift" and "clock" operators
    that naturally generalize the Pauli X and Z operators. These matrices span
    the entire space of `dim`-by-`dim` matrices as `k_1` and `k_2` range from 0
    to `dim-1`, inclusive.

    Note that the generalized Pauli operators are also known by the name of
    "discrete Weyl operators". [WatrousLec6]_

    Examples
    ==========

    The generalized Pauli operator for `k_1 = 1`, `k_2 = 0` and `dim = 2` is
    given as the standard Pauli-X matrix

    .. math::
        G_{1, 0, 2} = \begin{pmatrix}
                         0 & 1 \\
                         1 & 0
                      \end{pmatrix}.

    This can be obtained in `toqito` as follows.

    >>> from toqito.matrices import gen_pauli
    >>> dim = 2
    >>> k_1 = 1
    >>> k_2 = 0
    >>> gen_pauli(k_1, k_2, dim)
    [[0.+0.j, 1.+0.j],
     [1.+0.j, 0.+0.j]])

    The generalized Pauli matrix `k_1 = 1`, `k_2 = 1`, and `dim = 2` is given as
    the standard Pauli-Y matrix

    .. math::
        G_{1, 1, 2} = \begin{pmatrix}
                        0 & -1 \\
                        1 & 0
                      \end{pmatrix}.

    This can be obtained in `toqito` as follows.`

    >>> from toqito.matrices import gen_pauli
    >>> dim = 2
    >>> k_1 = 1
    >>> k_2 = 1
    >>> gen_pauli(k_1, k_2, dim)
    [[ 0.+0.0000000e+00j, -1.+1.2246468e-16j],
     [ 1.+0.0000000e+00j,  0.+0.0000000e+00j]])

    References
    ==========
    .. [WikGenPaul] Wikipedia: Generalizations of Pauli matrices
        https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices

    .. [WatrousLec6] Lecture 6: Further remarks on measurements and channels
        https://cs.uwaterloo.ca/~watrous/LectureNotes/CS766.Fall2011/06.pdf

    :param k_1: (a non-negative integer from 0 to `dim-1` inclusive).
    :param k_2: (a non-negative integer from 0 to `dim-1` inclusive).
    :param dim: (a positive integer indicating the dimension).
    :return: A generalized Pauli operator.
    """
    gen_pauli_x = shift(dim)
    gen_pauli_z = clock(dim)

    gen_pauli_w = np.linalg.matrix_power(gen_pauli_x, k_1) @ np.linalg.matrix_power(
        gen_pauli_z, k_2
    )

    return gen_pauli_w
