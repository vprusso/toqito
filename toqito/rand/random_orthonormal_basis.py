"""Generates a random orthonormal basis."""
import numpy as np

from toqito.rand import random_unitary


def random_orthonormal_basis(dim: int, is_real: bool = False, seed: int | None = None) -> list[np.ndarray]:
    r"""Generate a real random orthonormal basis of given dimension :math:`d`.

    The basis is generated from the columns of a random unitary matrix of the same dimension
    as the columns of a unitary matrix typically form an orthonormal basis :cite:`SE_1688950`.

    Examples
    ==========
    To generate a random orthonormal basis of dimension :math:`4`,

    >>> from toqito.rand import random_orthonormal_basis
    >>> random_orthonormal_basis(4, is_real = True) # doctest: +SKIP
    [array([ 0.18609797,  0.12416167,  0.28230062, -0.93287608]),
    array([-0.48238484,  0.72089168,  0.4763625 ,  0.14387088]),
    array([-0.66230111,  0.06548609, -0.6711639 , -0.32650855]),
    array([ 0.54224502,  0.67868302, -0.49287336,  0.04935131])]

    It is also possible to add a seed for reproducibility.

    >>> from toqito.rand import random_orthonormal_basis
    >>> random_orthonormal_basis(2, is_real=True, seed=42)
    [array([0.37621414, 0.92653274]), array([-0.92653274,  0.37621414])]


    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    dim: int
        Number of elements in the random orthonormal basis.
    seed: int | None
        A seed used to instantiate numpy's random number generator.

    """
    random_mat = random_unitary(dim, is_real, seed)
    return [random_mat[:, i] for i in range(dim)]
