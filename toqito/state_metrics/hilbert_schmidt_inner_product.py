"""Hilbert-Schmidt Inner Product."""
import numpy as np


def hilbert_schmidt_inner_product(A: np.ndarray, B: np.ndarray) -> float:
    r"""
    Compute the Hilbert-Schmidt inner product between two matrices [WikHSO].

    The Hilbert-Schmidt inner product between :math:`A` and :math:`B` is
    defined as

    .. math::
    
        HS = (A|B) = Tr[A^\dagger B]
    
    where :math:`|B) = vec(B)` and :math:`(A|` is the dual vector to :math:`|A)`.

    Examples
    ==========

    One may consider taking the Hilbert-Schmidt distance between the Hadamard matrix and Pauli-Z matrix

    >>> from toqito.matrices import hadamard
    >>> from toqito.matrices import pauli
    >>> h = hadamard(1)
    >>> pauli_z = pauli("Z")
    >>> hilbert_schmidt_inner_product(h, pauli_z)
    1

    References
    ==========
    .. [WikHSO] Wikipedia: Hilbert-Schmidt operator.
        https://en.wikipedia.org/wiki/Hilbert%E2%80%93Schmidt_operator

    :param A: An input matrix A.
    :param B: An input matrix B.
    :return: The Hilbert-Schmidt inner product between :code:`A` and :code:`B`.
    """
    hs_ip = np.trace(np.matmul(np.transpose(np.conj(A)), B))
    return np.ndarray.item(np.real_if_close(hs_ip, 1000))