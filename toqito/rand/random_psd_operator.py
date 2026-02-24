"""Generates a random positive semidefinite operator."""

import numpy as np


def random_psd_operator(
    dim: int,
    is_real: bool = False,
    seed: int | None = None,
    distribution: str = "uniform",
    scale: np.ndarray | None = None,
    num_degrees: int | None = None,
) -> np.ndarray:
    r"""Generate a random positive semidefinite operator.

    A positive semidefinite operator is a Hermitian operator that has only real and non-negative eigenvalues.
    This function generates a random PSD operator using one of two sampling strategies: ``"uniform"``
    constructs a Hermitian matrix via random sampling and eigendecomposition, while ``"wishart"`` samples
    from the Wishart distribution parameterized by a scale matrix and degrees of freedom.

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


    References
    ==========
    .. footbibliography::


    :param dim: The dimension of the operator.
    :param is_real: Boolean denoting whether the returned matrix will have all real entries or not.
                    Default is :code:`False`.
    :param seed: A seed used to instantiate numpy's random number generator.
    :return: A :code:`dim` x :code:`dim` random positive semidefinite matrix.
    :param distribution: The sampling strategy to use. Either ``"uniform"`` (default) or ``"wishart"``.
    :param scale: Scale matrix for the Wishart distribution. Defaults to the identity matrix if not provided.
                  Only used when ``distribution="wishart"``.
    :param num_degrees: Degrees of freedom for the Wishart distribution. Defaults to ``dim`` if not provided.
                        Only used when ``distribution="wishart"``.

    """
    # Generate a random matrix of dimension dim x dim.
    if not isinstance(dim, int) or dim < 0:
        raise ValueError("dim must be a non-negative integer.")

    gen = np.random.default_rng(seed=seed)

    if distribution == "uniform":
        rand_mat = gen.random((dim, dim))
        if not is_real:
            rand_mat = rand_mat + 1j * gen.random((dim, dim))
        rand_mat = (rand_mat.conj().T + rand_mat) / 2
        eigenvals, eigenvecs = np.linalg.eigh(rand_mat)
        q_mat, _ = np.linalg.qr(eigenvecs)
        return q_mat @ np.diag(np.abs(eigenvals)) @ q_mat.conj().T

    if distribution == "wishart":
        if scale is None:
            scale = np.eye(dim)
        if num_degrees is None:
            num_degrees = dim
        if is_real:
            x_mat = gen.multivariate_normal(np.zeros(dim), scale, size=num_degrees).T
        else:
            x_mat = (
                gen.multivariate_normal(np.zeros(dim), scale, size=num_degrees).T
                + 1j * gen.multivariate_normal(np.zeros(dim), scale, size=num_degrees).T
            )
        return x_mat @ x_mat.conj().T

    raise ValueError("Invalid distribution. Supported options are 'uniform' and 'wishart'.")
