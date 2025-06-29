"""Hilbert-Schmidt Inner Product refers to the inner product between two Hilbert-Schmidt operators."""

import numpy as np


def hilbert_schmidt_inner_product(a_mat: np.ndarray, b_mat: np.ndarray) -> complex:
    r"""Compute the Hilbert-Schmidt inner product between two matrices :footcite:`WikiHilbSchOp`.

    The Hilbert-Schmidt inner product between :code:`a_mat` and :code:`b_mat` is defined as

    .. math::

        HS = (A|B) = Tr[A^\dagger B]

    where :math:`|B\rangle = \text{vec}(B)` and :math:`\langle A|` is the dual vector to :math:`|A \rangle`.

    Note: This function has been adapted from :footcite:`Rigetti_2022_Forest`.

    Examples
    ==========

    One may consider taking the Hilbert-Schmidt distance between two Hadamard matrices.

    .. jupyter-execute::

     import numpy as np
     from toqito.matrices import hadamard
     from toqito.state_metrics import hilbert_schmidt_inner_product

     h = hadamard(1)

     np.around(hilbert_schmidt_inner_product(h, h), decimals=2)

    References
    ==========
    .. footbibliography::



    :param a_mat: An input matrix provided as a numpy array.
    :param b_mat: An input matrix provided as a numpy array.
    :return: The Hilbert-Schmidt inner product between :code:`a_mat` and
             :code:`b_mat`.

    """
    return np.trace(a_mat.conj().T @ b_mat)
