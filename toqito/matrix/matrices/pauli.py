"""Produces a Pauli operator."""
from typing import List, Union
from scipy import sparse
import numpy as np

from toqito.matrix.operations.tensor import tensor_list


def pauli(
    ind: Union[int, str, List[int], List[str]], is_sparse: bool = False
) -> Union[np.ndarray, sparse.csr_matrix]:
    """
    Produce a Pauli operator.

    Provides the 2-by-2 Pauli matrix indicated by the value of `ind`. The
    variable `ind = 1` gives the Pauli-X operator, `ind = 2` gives the Pauli-Y
    operator, `ind =3` gives the Pauli-Z operator, and `ind = 0`gives the
    identity operator. Alternatively, `ind` can be set to "I", "X", "Y", or "Z"
    (case insensitive) to indicate the Pauli identity, X, Y, or Z operator.

    References:
    [1] Wikipedia: Pauli matrices
        https://en.wikipedia.org/wiki/Pauli_matrices

    :param ind: The index to indicate which Pauli operator to generate.
    :param is_sparse: Returns a sparse matrix if set to True and a non-sparse
                      matrix if set to False.
    """
    if isinstance(ind, (int, str)):
        if ind in ("x", "X", 1):
            pauli_mat = np.array([[0, 1], [1, 0]])
        elif ind in ("y", "Y", 2):
            pauli_mat = np.array([[0, -1j], [1j, 0]])
        elif ind in ("z", "Z", 3):
            pauli_mat = np.array([[1, 0], [0, -1]])
        else:
            pauli_mat = np.identity(2)

        if is_sparse:
            pauli_mat = sparse.csr_matrix(pauli_mat)

        return pauli_mat

    num_qubits = len(ind)
    pauli_mats = []
    for i in range(num_qubits - 1, -1, -1):
        pauli_mats.append(pauli(ind[i], is_sparse))
    return tensor_list(pauli_mats)
