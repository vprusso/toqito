"""Generates a random positive semidefinite operator."""

import numpy as np


def random_psd_operator(
    dim: int,
    is_real: bool = False,
    seed: int | None = None,
    distribution: str = "uniform",
    scale: np.ndarray | None = None,
    df: int | None = None,
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
    :param distribution: Distribution used to generate the PSD operator.
                         Options are `"uniform"` (default) and `"wishart"`.
    :param scale: Scale matrix for Wishart distribution. Must be of shape
                  `(dim, dim)`. Defaults to the identity matrix.
    :param df: Degrees of freedom for Wishart distribution. Must be
               greater than or equal to `dim`. Defaults to `dim`.


    """
    # Generate a random matrix of dimension dim x dim.
    gen = np.random.default_rng(seed=seed)

    if not isinstance(dim, int) or dim <= 0:
        raise ValueError("`dim` must be a positive integer.")

    if distribution == "uniform":
        rand_mat = gen.random((dim, dim))

        if not is_real:
            rand_mat = rand_mat + 1j * gen.random((dim, dim))

        rand_mat = (rand_mat.conj().T + rand_mat) / 2
        eigenvals, eigenvecs = np.linalg.eigh(rand_mat)
        Q, _ = np.linalg.qr(eigenvecs)

        return Q @ np.diag(np.abs(eigenvals)) @ Q.conj().T

    elif distribution == "wishart":
        if scale is None:
            scale = np.eye(dim)

        if df is None:
            df = dim

        if not isinstance(df, int) or df < dim:
            raise ValueError("`df` must be an integer greater than or equal to `dim`.")

        if not isinstance(scale, np.ndarray):
            raise ValueError("`scale` must be a NumPy array.")

        if scale.shape != (dim, dim):
            raise ValueError("`scale` must have shape (dim, dim).")

        if is_real:
            X = gen.standard_normal((df, dim))
        else:
            X = gen.standard_normal((df, dim)) + 1j * gen.standard_normal((df, dim))

        return scale @ (X.conj().T @ X)

    else:
        raise ValueError("Invalid distribution. Supported options are 'uniform' and 'wishart'.")
