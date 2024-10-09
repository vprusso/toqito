"""Generate random orthonormal basis."""
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
    [array([0.52188745, 0.4983613 , 0.69049811, 0.04981832]),
    array([-0.48670459,  0.58756912, -0.10226756,  0.63829658]),
    array([ 0.23965404, -0.58538248,  0.187136  ,  0.75158061]),
    array([ 0.658269  ,  0.25243989, -0.69118291,  0.158815  ])]

    It is also possible to add a seed for reproducibility.

    >>> from toqito.rand import random_orthonormal_basis
    >>> random_orthonormal_basis(2, is_real=True, seed=42)
    [array([0.66954664, 0.74277002]), array([-0.74277002,  0.66954664])]


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
