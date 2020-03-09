"""Determines if a pure state is a product vector."""
from typing import List, Union
import numpy as np

from toqito.state.operations.schmidt_decomposition import schmidt_decomposition


def is_product_vector(vec: np.ndarray, dim: Union[int, List[int]]) -> bool:
    """
    Determine if a given vector is a product vector.

    :param vec: The vector to check.
    :param dim: The dimension of the vector
    :return: True if `vec` is a product vector and False otherwise.
    """
    eps = np.finfo(float).eps

    if dim is None:
        dim = np.round(np.sqrt(len(vec)))
    if isinstance(dim, list):
        dim = np.array(dim)

    # Allow the user to enter a single number for dim.
    if isinstance(dim, float):
        num_sys = 1
    else:
        num_sys = len(dim)

    if num_sys == 1:
        dim = np.array([dim, len(vec)/dim])
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * len(vec) * eps:
            raise ValueError("InvalidDim: The value of `dim` must evenly divide"
                             " `len(vec)`. Please provide a `dim` array "
                             "containing the dimensions of the subsystems.")
        dim[1] = np.round(dim[1])
        num_sys = 2

    # If there are only two subsystems, just use the Schmidt decomposition.
    if num_sys == 2:
        singular_vals, u_mat, vt_mat = schmidt_decomposition(vec, dim, 2)
        ipv = (singular_vals[1] <= np.prod(dim) * np.spacing(singular_vals[0]))
        # Provide this even if not requested, since it is needed if this
        # function was called as part of its recursive algorithm (see below)
        if ipv:
            u_mat = u_mat * np.sqrt(singular_vals[0])
            vt_mat = vt_mat * np.sqrt(singular_vals[0])
            dec = [u_mat[:, 0], vt_mat[:, 0]]
    else:
        new_dim = [dim[0] * dim[1]]
        new_dim.extend(dim[2:])
        ipv, dec = is_product_vector(vec, new_dim)
        if ipv:
            ipv, tdec = is_product_vector(dec[0], [dim[0], dim[1]])
            if ipv:
                dec = [tdec, dec[1:]]
    if not ipv:
        dec = 0

    return ipv, dec
