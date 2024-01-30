"""Pauli matrices."""


import numpy as np
from scipy import sparse

from toqito.matrix_ops import tensor


def pauli(
    ind: int | str | list[int] | list[str], is_sparse: bool = False
) -> np.ndarray | sparse.csr_matrix:
    r"""Produce a Pauli operator :cite:`WikiPauli`.

    Provides the 2-by-2 Pauli matrix indicated by the value of :code:`ind`. The
    variable :code:`ind = 1` gives the Pauli-X operator, :code:`ind = 2` gives
    the Pauli-Y operator, :code:`ind = 3` gives the Pauli-Z operator, and
    :code:`ind = 0` gives the identity operator. Alternatively, :code:`ind` can
    be set to "I", "X", "Y", or "Z" (case insensitive) to indicate the Pauli
    identity, X, Y, or Z operator.

    The 2-by-2 Pauli matrices are defined as the following matrices:

    .. math::

        \begin{equation}
            \begin{aligned}
                X = \begin{pmatrix}
                        0 & 1 \\
                        1 & 0
                    \end{pmatrix}, \quad
                Y = \begin{pmatrix}
                        0 & -i \\
                        i & 0
                    \end{pmatrix}, \quad
                Z = \begin{pmatrix}
                        1 & 0 \\
                        0 & -1
                    \end{pmatrix}, \quad
                I = \begin{pmatrix}
                        1 & 0 \\
                        0 & 1
                    \end{pmatrix}.
                \end{aligned}
            \end{equation}

    Examples
    ==========

    Example for identity Pauli matrix.

    >>> from toqito.matrices import pauli
    >>> pauli("I")
    [[1., 0.],
     [0., 1.]])

    Example for Pauli-X matrix.

    >>> from toqito.matrices import pauli
    >>> pauli("X")
    [[0, 1],
     [1, 0]])

    Example for Pauli-Y matrix.

    >>> from toqito.matrices import pauli
    >>> pauli("Y")
    [[ 0.+0.j, -0.-1.j],
     [ 0.+1.j,  0.+0.j]])

    Example for Pauli-Z matrix.

    >>> from toqito.matrices import pauli
    >>> pauli("Z")
    [[ 1,  0],
     [ 0, -1]])

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param ind: The index to indicate which Pauli operator to generate.
    :param is_sparse: Returns a sparse matrix if set to True and a non-sparse
                      matrix if set to False.

    """
    if isinstance(ind, (int, str)):
        if ind in {"x", "X", 1}:
            pauli_mat = np.array([[0, 1], [1, 0]])
        elif ind in {"y", "Y", 2}:
            pauli_mat = np.array([[0, -1j], [1j, 0]])
        elif ind in {"z", "Z", 3}:
            pauli_mat = np.array([[1, 0], [0, -1]])
        else:
            pauli_mat = np.identity(2)

        if is_sparse:
            pauli_mat = sparse.csr_matrix(pauli_mat)  # pylint: disable=redefined-variable-type

        return pauli_mat

    num_qubits = len(ind)
    pauli_mats = []
    for i in range(num_qubits - 1, -1, -1):
        pauli_mats.append(pauli(ind[i], is_sparse))
    return tensor(pauli_mats)
