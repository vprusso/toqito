"""Construct the ket object."""
import numpy as np


def ket(dim: int, pos: int) -> np.ndarray:
    """
    Obtain the ket of dimension `dim`.

    References:
    [1] https://en.wikipedia.org/wiki/Bra%E2%80%93ket_notation

    :param dim: The dimension of the column vector.
    :param pos: The position in which to place a 1.
    :return: The column vector of dimension `dim` with all entries set to `0`
             except the entry at position `1`.
    """
    if pos >= dim:
        msg = """
            InvalidDimension: The `pos` variable needs to be less than `dim`.        
        """
        raise ValueError(msg)

    ret = np.array(list(map(int, list(f"{0:0{dim}}"))))
    ret[pos] = 1
    ret = ret.conj().T.reshape(-1, 1)
    return ret
