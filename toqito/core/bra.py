"""Construct the bra object."""
import numpy as np


def bra(dim: int, pos: int) -> np.ndarray:
    r"""
    Obtain the bra of dimension `dim` [WIKBRA]_.

    Examples
    ==========

    The standard basis bra vectors given as :math:`\langle 0 |` and
    :math:`\langle 1 |` where

    .. math::
        \langle 0 | = \left[1, 0 \right] \quad \text{and} \quad \langle 1 | =
        \left[0, 1 \right],

    can be obtained in `toqito` as follows.

    Example:  Bra vector: :math:`\langle 0 |`.

    >>> from toqito.core.bra import bra
    >>> bra(2, 0)
    [1 0]

    Example:  Bra vector: :math:`\langle 1|`.

    >>> from toqito.core.bra import bra
    >>> bra(2, 1)
    [1 0]

    References
    ==========
    .. [WIKBRA] Wikipedia page for bra–ket notation:
           https://en.wikipedia.org/wiki/Bra%E2%80%93ket_notation

    :param dim: The dimension of the row vector.
    :param pos: The position in which to place a 1.
    :return: The row vector of dimension `dim` with all entries set to `0`
             except the entry at position `1`.
    """
    if pos >= dim:
        raise ValueError(
            "Invalid: The `pos` variable needs to be less than "
            "`dim` for bra function."
        )

    ret = np.array(list(map(int, list(f"{0:0{dim}}"))))
    ret[pos] = 1
    return ret
