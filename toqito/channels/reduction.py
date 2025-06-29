"""Generates the reduction channel."""

import numpy as np
from scipy.sparse import identity

from toqito.states import max_entangled


def reduction(dim: int, k: int = 1) -> np.ndarray:
    r"""Produce the reduction map or reduction channel :footcite:`WikiReductionCrit`.

    If :code:`k = 1`, this returns the Choi matrix of the reduction map which is a positive map
    on :code:`dim`-by-:code:`dim` matrices. For a different value of :code:`k`, this yields the
    Choi matrix of the map defined by:

    .. math::
        R(X) = k * \text{Tr}(X) * \mathbb{I} - X,

    where :math:`\mathbb{I}` is the identity matrix. This map is :math:`k`-positive.

    Examples
    ==========

    Using :code:`|toqito⟩`, we can generate the :math:`3`-dimensional (or standard) reduction map
    as follows.


    .. jupyter-execute::

     from toqito.channels import reduction

     reduction(3)

    References
    ==========
    .. footbibliography::


    :param dim: A positive integer (the dimension of the reduction map).
    :param k: If this positive integer is provided, the script will instead return the Choi
              matrix of the following linear map: Phi(X) := K * Tr(X)I - X.
    :return: The reduction map.

    """
    psi = max_entangled(dim, False, False)
    identity_matrix = identity(dim**2)
    return k * identity_matrix.toarray() - psi @ psi.conj().T
