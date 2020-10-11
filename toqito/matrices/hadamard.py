"""Hadamard matrix."""
import numpy as np


def hadamard(n_param: int = 1) -> np.ndarray:
    r"""
    Produce a 2^{n_param} dimensional Hadamard matrix [WikHad]_.

    The standard Hadamard matrix that is often used in quantum information as a
    two-qubit quantum gate is defined as

    .. math::
        H_1 = \frac{1}{\sqrt{2}} \begin{pmatrix}
                                    1 & 1 \\
                                    1 & -1
                                 \end{pmatrix}

    In general, the Hadamard matrix of dimension :math:2^{n_param} may be
    defined as

    .. math::
        \left( H_n \right)_{i, j} = \frac{1}{2^{\frac{n}{2}}}
        \left(-1\right)^{i \dot j}

    Examples
    ==========

    The standard 2-qubit Hadamard matrix can be generated in `toqito` as

    >>> from toqito.matrices import hadamard
    >>> hadamard(1)
    [[ 0.70710678  0.70710678]
     [ 0.70710678 -0.70710678]]

    References
    ==========
    .. [WikHad] Wikipedia: Hadamard transform
        https://en.wikipedia.org/wiki/Hadamard_transform

    :param n_param: A non-negative integer (default = 1).
    :return: The Hadamard matrix of dimension `2^{n_param}`.
    """
    if n_param == 0:
        return np.array([1])
    if n_param == 1:
        return 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    if n_param == 2:
        return 1 / 2 * np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
    if n_param > 1:
        mat_1 = hadamard(1)
        mat_2 = hadamard(1)
        mat = np.kron(mat_1, mat_2)
        for _ in range(2, n_param):
            mat_1 = mat_2
            mat_2 = mat
            mat = np.kron(mat_1, mat_2)
        return mat
    raise ValueError(f"Improper dimension {n_param} provided.")
