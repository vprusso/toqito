"""Generates the Pauli matrices."""

import numpy as np
from scipy.sparse import csr_array

from toqito.matrix_ops import tensor


def pauli(ind: int | str | list[int] | list[str], is_sparse: bool = False) -> np.ndarray | csr_array:
    r"""Produce a Pauli operator :cite:`WikiPauli`.

    Produces the 2-by-2 Pauli matrix indicated by the value of :code:`ind` or a tensor product
    of Pauli matrices when :code:`ind` is provided as a list. In general, when :code:`ind` is a list
    :math:`[i_1, i_2, \dots, i_n]`, the function returns the tensor product

    .. math:: P_{i_1} \otimes P_{i_2} \otimes \cdots \otimes P_{i_n}

    where each :math:`i_k \in \{0,1,2,3\}`, with the correspondence:
    :math:`P_{0} = I`, :math:`P_{1} = X`, :math:`P_{2} = Y`, and :math:`P_{3} = Z`.

    The 2-by-2 Pauli matrices are defined as follows:

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
    array([[1., 0.],
           [0., 1.]])

    Example for Pauli-X matrix.

    >>> from toqito.matrices import pauli
    >>> pauli("X")
    array([[0, 1],
           [1, 0]])

    Example for Pauli-Y matrix.

    >>> from toqito.matrices import pauli
    >>> pauli("Y")
    array([[ 0.+0.j, -0.-1.j],
           [ 0.+1.j,  0.+0.j]])

    Example for Pauli-Z matrix.

    >>> from toqito.matrices import pauli
    >>> pauli("Z")
    array([[ 1,  0],
           [ 0, -1]])

    Example using :math:`ind` as list.

    >>> from toqito.matrices import pauli
    >>> pauli([0,1])
    array([[0., 1., 0., 0.],
           [1., 0., 0., 0.],
           [0., 0., 0., 1.],
           [0., 0., 1., 0.]])


    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param ind: The index to indicate which Pauli operator to generate.
    :param is_sparse: Returns a compressed sparse row array if set to True and a non compressed
                      sparse row array if set to False.


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
            pauli_mat = csr_array(pauli_mat)

        return pauli_mat

    num_qubits = len(ind)
    pauli_mats = []
    for i in range(num_qubits):
        pauli_mats.append(pauli(ind[i], is_sparse))
    return tensor(pauli_mats)
