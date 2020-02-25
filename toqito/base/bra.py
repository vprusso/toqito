"""Construct the bra object."""
import numpy as np


def bra(dim: int, pos: int) -> np.ndarray:
    """
    Obtain the bra of dimension `dim`.

    References:
    [1] Wikipedia page for bra–ket notation:
        https://en.wikipedia.org/wiki/Bra%E2%80%93ket_notation

    :param dim: The dimension of the row vector.
    :param pos: The position in which to place a 1.
    :return: The row vector of dimension `dim` with all entries set to `0`
             except the entry at position `1`.
    """
    if pos >= dim:
        raise ValueError("Invalid: The `pos` variable needs to be less than "
                         "`dim` for bra function.")

    ret = np.array(list(map(int, list(f"{0:0{dim}}"))))
    ret[pos] = 1
    return ret
