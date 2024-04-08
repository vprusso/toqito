"""Generate random unitary."""

import numpy as np


def random_unitary(dim: list[int] | int, is_real: bool = False) -> np.ndarray:
    """Generate a random unitary or orthogonal matrix :cite:`Ozols_2009_RandU`.

    Calculates a random unitary matrix (if :code:`is_real = False`) or a random real orthogonal
    matrix (if :code:`is_real = True`), uniformly distributed according to the Haar measure.

    Examples
    ==========

    We may generate a random unitary matrix. Here is an example of how we may be able to generate a
    random :math:`2`-dimensional random unitary matrix with complex entries.

    >>> from toqito.rand import random_unitary
    >>> complex_dm = random_unitary(2)
    >>> complex_dm # doctest: +SKIP
    array([[ 0.13764463+0.65538975j,  0.74246453+0.01626838j],
           [ 0.45776527+0.58478132j, -0.6072508 +0.28236187j]])


    We can verify that this is in fact a valid unitary matrix using the :code:`is_unitary` function
    from :code:`toqito` as follows

    >>> from toqito.matrix_props import is_unitary
    >>> is_unitary(complex_dm)
    True

    We can also generate random unitary matrices that are real-valued as follows.

    >>> from toqito.rand import random_unitary
    >>> real_dm = random_unitary(2, True)
    >>> real_dm # doctest: +SKIP
    array([[ 0.87766506, -0.47927449],
           [ 0.47927449,  0.87766506]])


    Again, verifying that this is a valid unitary matrix can be done as follows.

    >>> from toqito.matrix_props import is_unitary
    >>> is_unitary(real_dm)
    True

    We may also generate unitaries such that the dimension argument provided is a :code:`list` as
    opposed to an :code:`int`. Here is an example of a random unitary matrix of dimension :math:`4`.

    >>> from toqito.rand import random_unitary
    >>> mat = random_unitary([4, 4], True)
    >>> mat # doctest: +SKIP
    array([[ 0.49527332,  0.08749933, -0.16968586,  0.84749922],
           [ 0.68834418, -0.26695275,  0.62674543, -0.24921614],
           [ 0.38614979, -0.438767  , -0.7417619 , -0.32887862],
           [ 0.36300822,  0.85355938, -0.16788735, -0.33387909]])


    As before, we can verify that this matrix generated is a valid unitary matrix.

    >>> from toqito.matrix_props import is_unitary
    >>> is_unitary(mat)
    True

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :param dim: The number of rows (and columns) of the unitary matrix.
    :param is_real: Boolean denoting whether the returned matrix has real
                    entries or not. Default is :code:`False`.
    :return: A :code:`dim`-by-:code:`dim` random unitary matrix.

    """
    if isinstance(dim, int):
        dim = [dim, dim]

    if dim[0] != dim[1]:
        raise ValueError("Unitary matrix must be square.")

    # Construct the Ginibre ensemble.
    gin = np.random.rand(dim[0], dim[1])

    if not is_real:
        gin = gin + 1j * np.random.rand(dim[0], dim[1])

    # QR decomposition of the Ginibre ensemble.
    q_mat, r_mat = np.linalg.qr(gin)

    # Compute U from QR decomposition.
    r_mat = np.sign(np.diag(r_mat))

    # Protect against potentially zero diagonal entries.
    r_mat[r_mat == 0] = 1

    return q_mat @ np.diag(r_mat)
