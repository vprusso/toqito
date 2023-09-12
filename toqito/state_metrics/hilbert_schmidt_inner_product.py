"""Hilbert-Schmidt Inner Product."""
import numpy as np


def hilbert_schmidt_inner_product(a_mat: np.ndarray, b_mat: np.ndarray) -> complex:
    r"""
    Compute the Hilbert-Schmidt inner product between two matrices [WikHSO]_.

    The Hilbert-Schmidt inner product between :code:`a_mat` and :code:`b_mat` is
    defined as

    .. math::

        HS = (A|B) = Tr[A^\dagger B]

    where :math:`|B\rangle = \text{vec}(B)` and :math:`\langle A|` is the dual
    vector to :math:`|A \rangle`.

    Note: This function has been adapted from [Rigetti21]_.

    Examples
    ==========

    One may consider taking the Hilbert-Schmidt distance between two Hadamard
    matrices.

    >>> from toqito.matrices import hadamard
    >>> from toqito.matrices import pauli
    >>> h = hadamard(1)
    >>> hilbert_schmidt_inner_product(h, h)
    2

    References
    ==========
    .. [WikHSO] Wikipedia: Hilbert-Schmidt operator.
        https://en.wikipedia.org/wiki/Hilbert%E2%80%93Schmidt_operator
    .. [Rigetti21] Forest Benchmarking (Rigetti).
        https://github.com/rigetti/forest-benchmarking

    :param a_mat: An input matrix provided as a numpy array.
    :param b_mat: An input matrix provided as a numpy array.
    :return: The Hilbert-Schmidt inner product between :code:`a_mat` and
             :code:`b_mat`.
    """
    return np.trace(a_mat.conj().T @ b_mat)
