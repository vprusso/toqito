"""Permutation operator is a unitary operator that permutes subsystems."""

import numpy as np
import scipy as sp

from toqito.perms import permute_systems


def permutation_operator(
    dim: list[int] | int,
    perm: list[int],
    inv_perm: bool = False,
    is_sparse: bool = False,
) -> np.ndarray:
    r"""Produce a unitary operator that permutes subsystems.

    Generates a unitary operator that permutes the order of subsystems according to the permutation vector :code:`perm`,
    where the :math:`i^{th}` subsystem has dimension :code:`dim[i]`.

    If :code:`inv_perm` = True, it implements the inverse permutation of :code:`perm`. The permutation operator return
    is full is :code:`is_sparse` is :code:`False` and sparse if :code:`is_sparse` is :code:`True`.

    Examples
    ==========

    The permutation operator obtained with dimension :math:`d = 2` is equivalent to the standard swap operator on two
    qubits

    .. math::
        P_{2, [1, 0]} =
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1
        \end{pmatrix}

    Using :code:`toqito`, this can be achieved in the following manner.

    >>> from toqito.perms import permutation_operator
    >>> permutation_operator(2, [1, 0])
    array([[1., 0., 0., 0.],
           [0., 0., 1., 0.],
           [0., 1., 0., 0.],
           [0., 0., 0., 1.]])

    :param dim: The dimensions of the subsystems to be permuted.
    :param perm: A permutation vector.
    :param inv_perm: Boolean dictating if :code:`perm` is inverse or not.
    :param is_sparse: Boolean indicating if return is sparse or not.
    :return: Permutation operator of dimension :code:`dim`.

    """
    # Allow the user to enter a single number for `dim`.
    if isinstance(dim, int):
        dim = [dim] * np.ones(max(perm) + 1)
    if isinstance(dim, list):
        dim = np.array(dim)

    mat = sp.sparse.identity(int(np.prod(dim))) if is_sparse else np.identity(int(np.prod(dim)))
    # Swap the rows of the identity matrix appropriately.

    return permute_systems(mat, perm, dim, True, inv_perm)
