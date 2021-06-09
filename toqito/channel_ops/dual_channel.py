"""Compute the dual of a map"""
from typing import List, Union
import numpy as np

from toqito.matrix_props import is_square
from toqito.perms import swap

def dual_channel(
    phi_op: Union[np.ndarray, List[List[np.ndarray]]],
    dims: List[int] = None
) -> Union[np.ndarray, List[List[np.ndarray]]]:
    r"""
    TODO: docstring
    """
    # If phi_op is a list, assume it contains couples of Kraus operators
    # and take the Hermitian conjugate
    if isinstance(phi_op, list):
        return [[a.conj().T for a in x] for x in phi_op]

    # If phi_op is a ndarray, assume it is a Choi matrix
    if isnstance(phi_op, np.ndarray):
        if not(is_square(phi_op)):
            raise ValueError("Invalid: `phi_op` is not a valid Choi matrix (not square).")
        if dims is None:
            sqr = np.sqrt(phi_op.shape[0])
            if sqr.is_integer():
                dims = [int(round(sqr))] * 2
            else:
                raise ValueError(
                    "The dimensions `dims` of the input and output spaces should be specified."
                    )
        return swap(phi_op.conj(), dim=dims)
    raise ValueError(
        "Invalid: The variable `phi_op` must either be a list of "
        "Kraus operators or as a Choi matrix."
    )
