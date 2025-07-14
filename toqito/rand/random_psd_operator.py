"""Generates a random positive semidefinite operator."""

import numpy as np


def random_psd_operator(
    dim: int,
    is_real: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    r"""Generate a random positive semidefinite operator.

    A positive semidefinite operator is a Hermitian operator that has only real and non-negative eigenvalues.
    This function generates a random positive semidefinite operator by constructing a Hermitian matrix,
    based on the fact that a Hermitian matrix can have real eigenvalues.

    Examples
    ========

    Using :code:`|toqito⟩`, we may generate a random positive semidefinite matrix.
    For :math:`\text{dim}=2`, this can be accomplished as follows.

    .. jupyter-execute::

     from toqito.rand import random_psd_operator

     complex_psd_mat = random_psd_operator(2)

     complex_psd_mat

    We can confirm that this matrix indeed represents a valid positive semidefinite matrix by utilizing
    the :code:`is_positive_semidefinite` function from the :code:`|toqito⟩` library, as demonstrated below:

    .. jupyter-execute::

     from toqito.matrix_props import is_positive_semidefinite

     is_positive_semidefinite(complex_psd_mat)


    We can also generate random positive semidefinite matrices that are real-valued as follows.

    .. jupyter-execute::

     from toqito.rand import random_psd_operator

     real_psd_mat = random_psd_operator(2, is_real=True)

     real_psd_mat


    Again, verifying that this is a valid positive semidefinite matrix can be done as follows.

    .. jupyter-execute::

     from toqito.matrix_props import is_positive_semidefinite
     is_positive_semidefinite(real_psd_mat)


    It is also possible to add a seed for reproducibility.

    .. jupyter-execute::

     from toqito.rand import random_psd_operator

     seeded = random_psd_operator(2, is_real=True, seed=42)

     seeded


    References
    ==========
    .. footbibliography::


    :param dim: The dimension of the operator.
    :param is_real: Boolean denoting whether the returned matrix will have all real entries or not.
                    Default is :code:`False`.
    :param seed: A seed used to instantiate numpy's random number generator.
    :return: A :code:`dim` x :code:`dim` random positive semidefinite matrix.

    """
    # Generate a random matrix of dimension dim x dim.
    gen = np.random.default_rng(seed=seed)
    rand_mat = gen.random((dim, dim))

    # If is_real is False, add an imaginary component to the matrix.
    if not is_real:
        rand_mat = rand_mat + 1j * gen.random((dim, dim))

    # Constructing a Hermitian matrix.
    rand_mat = (rand_mat.conj().T + rand_mat) / 2
    eigenvals, eigenvecs = np.linalg.eigh(rand_mat)

    Q, R = np.linalg.qr(eigenvecs)

    return Q @ np.diag(np.abs(eigenvals)) @ Q.conj().T
