"""Produces a dephasing channel."""
import numpy as np
from toqito.states.states.max_entangled import max_entangled


def dephasing(dim: int, param_p: int = 0) -> np.ndarray:
    r"""
    Produce the dephasing channel.

    The dephasing channel is the Choi matrix of the completely dephasing
    channel that acts on `dim`-by-`dim` matrices.

    Produces the partially dephasing channel `(1-P)*D + P*ID` where `D` is the
    completely dephasing channel and ID is the identity channel.

    Examples
    ==========

    The dephasing channel maps kills everything off the diagonals. Consider the
    following matrix

    .. math::
        \rho = \begin{pmatrix}
                   1 & 2 & 3 & 4 \\
                   5 & 6 & 7 & 8 \\
                   9 & 10 & 11 & 12 \\
                   13 & 14 & 15 & 16
               \end{pmatrix}.

    Applying the dephasing channel to :math:`\rho` we have that

    .. math::
        \Phi(\rho) = \begin{pmatrix}
                         1 & 0 & 0 & 0 \\
                         0 & 6 & 0 & 0 \\
                         0 & 0 & 11 & 0 \\
                         0 & 0 & 0 & 16
                     \end{pmatrix}.

    This can be observed in `toqito` as follows.

    >>> from toqito.maps.apply_map import apply_map
    >>> from toqito.channels.dephasing import dephasing
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> apply_map(test_input_mat, dephasing(4))
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  6.,  0.,  0.],
           [ 0.,  0., 11.,  0.],
           [ 0.,  0.,  0., 16.]])

    :param dim: The dimensionality on which the channel acts.
    :param param_p: Default is 0.
    :return:
    """
    # Compute the Choi matrix of the dephasing channel.

    # Gives a sparse non-normalized state.
    psi = max_entangled(dim=dim, is_sparse=False, is_normalized=False)
    return (1 - param_p) * np.diag(np.diag(psi * psi.conj().T)) + param_p * (
        psi * psi.conj().T
    )
