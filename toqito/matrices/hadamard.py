"""Hadamard matrix."""
import numpy as np


def hadamard(n_param: int = 1) -> np.ndarray:
    r"""Produce a :code:`2^{n_param}` dimensional Hadamard matrix :cite:`WikiHadamard`.

    The standard Hadamard matrix that is often used in quantum information as a
    two-qubit quantum gate is defined as

    .. math::
        H_1 = \frac{1}{\sqrt{2}} \begin{pmatrix}
                                    1 & 1 \\
                                    1 & -1
                                 \end{pmatrix}

    In general, the Hadamard matrix of dimension :code:`2^{n_param}` may be
    defined as

    .. math::
        \left( H_n \right)_{i, j} = \frac{1}{2^{\frac{n}{2}}}
        \left(-1\right)^{i \dot j}

    Examples
    ==========

    The standard 2-qubit Hadamard matrix can be generated in :code:`toqito` as

    >>> from toqito.matrices import hadamard
    >>> hadamard(1)
    [[ 0.70710678  0.70710678]
     [ 0.70710678 -0.70710678]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames



    :param n_param: A non-negative integer (default = 1).
    :return: The Hadamard matrix of dimension :code:`2^{n_param}`.

    """
    return 2 ** (-n_param / 2) * np.array(
        [
            [(-1) ** _hamming_distance(i & j) for i in range(2 ** n_param)]
            for j in range(2 ** n_param)
        ]
    )


def _hamming_distance(x_param: int) -> int:
    """Calculate the bit-wise Hamming distance of :code:`x_param` from 0.

    The Hamming distance is the number 1s in the integer :code:`x_param`.

    :param x_param: A non-negative integer.
    :return: The hamming distance of :code:`x_param` from 0.
    """
    tot = 0
    while x_param:
        tot += 1
        x_param &= x_param - 1
    return tot
