"""Construct standard basis."""
import numpy as np


def standard_basis(dim: int, flatten: bool = False) -> list[np.ndarray]:
    """Create standard basis of dimension :code:`dim`.

    Create a list containing the elements of the standard basis for the
    given dimension:

    .. math::

        |1> = (1, 0, 0, ..., 0)^T
        |2> = (0, 1, 0, ..., 0)^T
        .
        .
        .
        |n> = (0, 0, 0, ..., 1)^T

    This function was inspired by :cite:`Seshadri_2021_Git, Seshadri_2021_Theory, Seshadri_2021_Versatile`

    Examples
    ========
    >>> from toqito.matrices import standard_basis
    >>> standard_basis(2)
    [array([[1.],
            [0.]]),
    array([[0.],
            [1.]])]


    References
    ==========
    .. bibliography::
        :filter: docname in docnames

    :param dim: The dimension of the basis.
    :param flatten: If True, the basis is returned as a flattened list.
    :return: A list of numpy.ndarray of shape (n, 1).

    """
    first_basis_vector = np.zeros(dim) if flatten else np.zeros((dim, 1))
    first_basis_vector[0] = 1.0

    # The standard_basis is obtained by cyclic permutations of the first basis
    # vector
    return [np.array([first_basis_vector[i - j] for i in range(dim)]) for j in range(dim)]
