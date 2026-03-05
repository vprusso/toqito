"""Generates a random positive semidefinite operator."""

import warnings

import numpy as np

from toqito.matrix_props import is_positive_semidefinite


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
    This function generates a random PSD operator using one of two sampling strategies: `"uniform"`
    constructs a Hermitian matrix via random sampling and eigendecomposition, while `"wishart"` samples
    from the Wishart distribution parameterized by a scale matrix and degrees of freedom.

    Examples:
        Using `|toqito⟩`, we may generate a random positive semidefinite matrix.
        For \(\text{dim}=2\), this can be accomplished as follows.

        ```python exec="1" source="above" session="psd_operator"
        from toqito.rand import random_psd_operator

        complex_psd_mat = random_psd_operator(2)

        print(complex_psd_mat)
        ```

        We can confirm that this matrix indeed represents a valid positive semidefinite matrix by utilizing
        the `is_positive_semidefinite` function from the `|toqito⟩` library, as demonstrated below:

        ```python exec="1" source="above" session="psd_operator"
        from toqito.matrix_props import is_positive_semidefinite

        print(is_positive_semidefinite(complex_psd_mat))
        ```


        We can also generate random positive semidefinite matrices that are real-valued as follows.

        ```python exec="1" source="above" session="psd_operator"
        real_psd_mat = random_psd_operator(2, is_real=True)

        print(real_psd_mat)
        ```


        Again, verifying that this is a valid positive semidefinite matrix can be done as follows.

        ```python exec="1" source="above" session="psd_operator"
        print(is_positive_semidefinite(real_psd_mat))
        ```


        It is also possible to add a seed for reproducibility.

        ```python exec="1" source="above" session="psd_operator"
        seeded = random_psd_operator(2, is_real=True, seed=42)

        print(seeded)
        ```

        To generate a random PSD operator using the Wishart distribution, pass
        `distribution="wishart"`. Optional parameters `scale` (a PSD scale matrix) and
        `num_degrees` (degrees of freedom) can be provided to fully parameterize the distribution.

        ```python exec="1" source="above" session="psd_operator"
        wishart_mat = random_psd_operator(3, distribution="wishart", num_degrees=5)

        print(wishart_mat)
        ```

    Args:
        dim: The dimension of the operator.
        is_real: Boolean denoting whether the returned matrix will have all real entries or not.
            Default is `False`.
        seed: A seed used to instantiate numpy's random number generator.
        distribution: The sampling strategy to use. Either `"uniform"` (default) or
            `"wishart"`. The `"uniform"` strategy constructs a Hermitian matrix via
            random sampling and eigendecomposition. The `"wishart"` strategy samples from
            the Wishart distribution, which guarantees a PSD matrix by construction.
        scale: Scale matrix for the Wishart distribution. Must be a positive semidefinite matrix of
            shape `(dim, dim)`. Defaults to the identity matrix if not provided.
            Only used when `distribution="wishart"`.
        num_degrees: Degrees of freedom for the Wishart distribution. Must be a positive integer.
            Defaults to `dim` if not provided. Only used when `distribution="wishart"`.
            When `num_degrees < dim`, the resulting matrix is guaranteed to be rank-deficient
            (singular), which may be unintentional.

    Returns:
        A `dim` x `dim` random positive semidefinite matrix.

    """
    if not isinstance(dim, int) or dim < 1:
        raise ValueError("dim must be a positive integer.")

    if distribution == "uniform":
        if scale is not None or num_degrees is not None:
            warnings.warn(
                "scale and num_degrees are ignored when distribution='uniform'.",
                UserWarning,
                stacklevel=2,
            )
        gen = np.random.default_rng(seed=seed)
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
        else:
            if scale.shape != (dim, dim):
                raise ValueError(f"scale must be a {dim}x{dim} matrix, got {scale.shape}.")
            if not is_positive_semidefinite(scale):
                raise ValueError("scale must be a positive semidefinite matrix.")

        if num_degrees is None:
            num_degrees = dim
        if num_degrees < 1:
            raise ValueError("num_degrees must be a positive integer.")

        if num_degrees < dim:
            warnings.warn(
                f"num_degrees ({num_degrees}) < dim ({dim}): the resulting Wishart matrix "
                "will be rank-deficient (singular).",
                UserWarning,
                stacklevel=2,
            )

        gen = np.random.default_rng(seed=seed)
        if is_real:
            x_mat = gen.multivariate_normal(np.zeros(dim), scale, size=num_degrees).T
        else:
            # Each component is drawn independently with covariance `scale`.
            # Dividing by sqrt(2) ensures the resulting complex Wishart matrix
            # x_mat @ x_mat.conj().T has the expected scale matrix `scale`
            # rather than 2 * scale.
            x_mat = (
                gen.multivariate_normal(np.zeros(dim), scale, size=num_degrees).T
                + 1j * gen.multivariate_normal(np.zeros(dim), scale, size=num_degrees).T
            ) / np.sqrt(2)
        return x_mat @ x_mat.conj().T

    raise ValueError("Invalid distribution. Supported options are 'uniform' and 'wishart'.")
