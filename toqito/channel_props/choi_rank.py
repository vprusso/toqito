"""Calculates the Choi rank of a channel."""

import numpy as np

from toqito.channel_ops import kraus_to_choi


def choi_rank(phi: np.ndarray | list[list[np.ndarray]]) -> int:
    r"""Calculate the rank of the Choi representation of a quantum channel.

    (Section 2.2: Quantum Channels from :cite:`Watrous_2018_TQI`).

    Examples
    ==========

    The transpose map can be written either in Choi representation (as a
    SWAP operator) or in Kraus representation. If we choose the latter, it
    will be given by the following matrices:

    .. math::
        \begin{equation}
            \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                0 & i \\ -i & 0
            \end{pmatrix}, \quad
            \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                0 & 1 \\
                1 & 0
            \end{pmatrix}, \quad
            \begin{pmatrix}
                1 & 0 \\
                0 & 0
            \end{pmatrix}, \quad
            \begin{pmatrix}
                0 & 0 \\
                0 & 1
            \end{pmatrix}.
        \end{equation}

    and can be generated in :code:`toqito` with the following list:

    >>> import numpy as np
    >>> kraus_1 = np.array([[1, 0], [0, 0]])
    >>> kraus_2 = np.array([[1, 0], [0, 0]]).conj().T
    >>> kraus_3 = np.array([[0, 1], [0, 0]])
    >>> kraus_4 = np.array([[0, 1], [0, 0]]).conj().T
    >>> kraus_5 = np.array([[0, 0], [1, 0]])
    >>> kraus_6 = np.array([[0, 0], [1, 0]]).conj().T
    >>> kraus_7 = np.array([[0, 0], [0, 1]])
    >>> kraus_8 = np.array([[0, 0], [0, 1]]).conj().T
    >>> kraus_ops = [[kraus_1, kraus_2], [kraus_3, kraus_4],[kraus_5, kraus_6],[kraus_7, kraus_8]]

    To calculate its Choi rank, we proceed in the following way:

    >>> from toqito.channel_props import choi_rank
    >>> choi_rank(kraus_ops)
    np.int64(4)

    We can the verify the associated Choi representation (the SWAP gate)
    gets the same Choi rank:

    >>> choi_matrix = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    >>> choi_rank(choi_matrix)
    np.int64(4)

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :raises ValueError: If matrix is not Choi.
    :param phi: Either a Choi matrix or a list of Kraus operators
    :return: The Choi rank of the provided channel representation.

    """
    if isinstance(phi, list):
        phi = kraus_to_choi(phi)
    elif not isinstance(phi, np.ndarray):
        raise ValueError("Not a valid Choi matrix.")

    return np.linalg.matrix_rank(phi)
