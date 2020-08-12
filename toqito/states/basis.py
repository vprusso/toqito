"""Basis state."""
import numpy as np


def basis(dim: int, pos: int) -> np.ndarray:
    r"""

    Obtain the ket of dimension :code:`dim` [WikKet]_.

    Examples
    ==========

    The standard basis bra vectors given as :math:`|0 \rangle` and :math:`|1 \rangle` where

    .. math::
        |0 \rangle = \left[1, 0 \right]^{\text{T}} \quad \text{and} \quad
        |1 \rangle = \left[0, 1 \right]^{\text{T}},

    can be obtained in :code:`toqito` as follows.

    Example:  Ket basis vector: :math:`|0\rangle`.

    >>> from toqito.states import basis
    >>> basis(2, 0)
    [[1]
    [0]]

    Example: Ket basis vector: :math:`|1\rangle`.

    >>> from toqito.states import basis
    >>> basis(2, 1)
    [[0]
    [1]]

    References
    ==========
    .. [WikKet] Wikipedia page for bra–ket notation:
           https://en.wikipedia.org/wiki/Bra%E2%80%93ket_notation

    :param dim: The dimension of the column vector.
    :param pos: The position in which to place a 1.
    :return: The column vector of dimension :code:`dim` with all entries set to `0` except the entry
             at position `1`.
    """
    if pos >= dim:
        raise ValueError(
            "Invalid: The `pos` variable needs to be less than `dim` for ket function."
        )

    ret = np.array(list(map(int, list(f"{0:0{dim}}"))))
    ret[pos] = 1
    ret = ret.conj().T.reshape(-1, 1)
    return ret
