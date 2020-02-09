"""Construct the ket object."""
import numpy as np


def ket(dim: int, pos: int) -> np.ndarray:
    """
    :param dim:
    :param pos:
    :return:
    """
    ret = np.array(list(map(int, list(f"{0:0{dim}}"))))
    ret[pos] = 1
    ret = ret.conj().T.reshape(-1, 1)
    return ret
