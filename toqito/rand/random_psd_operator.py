"""Generates a random positive semidefinite operator."""

import numpy as np


def random_psd_operator(
    dim: int,
    is_real: bool = False,
) -> np.ndarray:
    r"""Generate a random positive semidefinite operator.

    A positive semidefinite operator is a Hermitian operator that has only real and non-negative eigenvalues.
    This function generates a random positive semidefinite operator by constructing a Hermitian matrix,
    based on the fact that a Hermitian matrix can have real eigenvalues.

    Examples
    ===========================

    Using :code:`toqito`, we may generate a random positive semidefinite matrix.
    For :math:`dim=2`, this can be accomplished as follows.

    >>> from toqito.rand import random_psd_operator
    >>> complex_psd_mat = random_psd_operator(2)
    >>> complex_psd_mat # doctest: +SKIP
    array([[0.42313949+3.85185989e-34j, 0.35699744-1.81934920e-02j],
           [0.35699744+1.81934920e-02j, 0.36668881+0.00000000e+00j]])

    We can confirm that this matrix indeed represents a valid positive semidefinite matrix by utilizing
    the :code: `is_positive_semidefinite`
    function from the :code: `toqito` library, as demonstrated below:

    >>> from toqito.matrix_props import is_positive_semidefinite
    >>> is_positive_semidefinite(complex_psd_mat)
    True

    We can also generate random positive semidefinite matrices that are real-valued as follows.

    >>> from toqito.rand import random_density_matrix
    >>> real_psd_mat = random_density_matrix(2, is_real=True)
    >>> real_psd_mat # doctest: +SKIP
    array([[0.68112055, 0.14885971],
           [0.14885971, 0.62955916]])

    Again, verifying that this is a valid positive semidefinite matrix can be done as follows.

    >>> from toqito.matrix_props import is_positive_semidefinite
    >>> is_positive_semidefinite(real_psd_mat)
    True

    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param dim: The dimension of the operator.
    :param is_real: Boolean denoting whether the returned matrix will have all real entries or not.
                    Default is :code:`False`.
    :return: A :code:`dim`-by-:code:`dim` random positive semidefinite matrix.

    """
    # Generate a random matrix of dimension dim x dim.
    rand_mat = np.random.rand(dim, dim)

    # If is_real is False, add an imaginary component to the matrix.
    if not is_real:
        rand_mat = rand_mat + 1j * np.random.rand(dim, dim)

    # Constructing a Hermitian matrix.
    rand_mat = (rand_mat.conj().T + rand_mat) / 2
    eigenvals, eigenvecs = np.linalg.eigh(rand_mat)

    # Constructing a positive semidefinite matrix.
    Q, R = np.linalg.qr(eigenvecs)
    psd_matrix = Q @ np.diag(np.abs(eigenvals)) @ Q.conj().T

    return psd_matrix
