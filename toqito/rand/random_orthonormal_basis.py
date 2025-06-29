"""Generates a random orthonormal basis."""

import numpy as np

from toqito.rand import random_unitary


def random_orthonormal_basis(dim: int, is_real: bool = False, seed: int | None = None) -> list[np.ndarray]:
    r"""Generate a real random orthonormal basis of given dimension :math:`d`.

    The basis is generated from the columns of a random unitary matrix of the same dimension
    as the columns of a unitary matrix typically form an orthonormal basis :footcite:`SE_1688950`.

    Examples
    ==========
    To generate a random orthonormal basis of dimension :math:`4`,

    .. jupyter-execute::

     from toqito.rand import random_orthonormal_basis

     random_orthonormal_basis(4, is_real = True)

    It is also possible to add a seed for reproducibility.

    .. jupyter-execute::

     from toqito.rand import random_orthonormal_basis

     random_orthonormal_basis(2, is_real=True, seed=42)


    References
    ==========
    .. footbibliography::


    dim: int
        Number of elements in the random orthonormal basis.
    seed: int | None
        A seed used to instantiate numpy's random number generator.

    """
    random_mat = random_unitary(dim, is_real, seed)
    return [random_mat[:, i] for i in range(dim)]
