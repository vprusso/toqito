"""Computes the Schmidt rank of a bipartite vector."""
from typing import List, Union
import numpy as np


def schmidt_rank(vec: np.ndarray,
                 dim: Union[int, List[int], np.ndarray] = None,
                 tol: float = None) -> float:
    """
    Compute the Schmidt rank.

    Compute the Schmidt rank of the vector `vec`, assumed to live in bipartite
    space, where both subsystems have dimension equal to `sqrt(len(vec))`.

    The dimension may be specified by the 1-by-2 vector `dim` and the rank in
    that case is determined as the number of Schmidt coefficients larger than
    `tol`.

    :param vec: A bipartite vector to have its Schmidt rank computed.
    :param dim: A 1-by-2 vector.
    :param tol: The tolerance parameter for rank calculation.
    :return: The Schmidt rank of vector `vec`.
    """
    eps = np.finfo(float).eps
    slv = np.round(np.sqrt(len(vec)))

    if dim is None:
        dim = slv
    if tol is None:
        tol = eps

    if isinstance(dim, int):
        dim = np.array([dim, len(vec)/dim])
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * len(vec) * eps:
            raise ValueError("Invalid: The value of `dim` must evenly divide "
                             "`len(vec)`; please provide a `dim` array "
                             "containing the dimensions of the subsystems")
        dim[1] = np.round(dim[1])

    return np.linalg.matrix_rank(vec, dim[::-1], tol)
