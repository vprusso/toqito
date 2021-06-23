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
    Compute the dual of a map (quantum channel) [WatDChan18]_ .

    References
    ==========
    .. [WatDChan18] Watrous, John.
        The theory of quantum information.
        Section: Representations and characterizations of channels.
        Cambridge University Press, 2018.

    :param phi_op: A superoperator. :code:`phi_op` should be provided either as a Choi matrix,
                   or as a list of numpy arrays with 2 columns whose entries are its
                   Kraus operators.
    :param dims: Dimension of the input and output systems, for Choi matrix representation.
                 If :code:`None`, try to infer them from :code:`phi_op.shape`.
    :return: The map dual to :chode:`phi_op`, in the same representation.
    """
    # If phi_op is a list, assume it contains couples of Kraus operators
    # and take the Hermitian conjugate
    if isinstance(phi_op, list):
        if isinstance(phi_op[0], list):
            return [[a.conj().T for a in x] for x in phi_op]
        if isinstance(phi_op[0], np.ndarray):
            return [a.conj().T for a in phi_op]

    # If phi_op is a ndarray, assume it is a Choi matrix
    if isinstance(phi_op, np.ndarray):
        if len(phi_op.shape) == 2:
            if not is_square(phi_op):
                raise ValueError("Invalid: `phi_op` is not a valid Choi matrix (not square).")
            if dims is None:
                sqr = np.sqrt(phi_op.shape[0])
                if sqr.is_integer():
                    dims = [int(round(sqr))] * 2
                else:
                    raise ValueError(
                        "The dimensions `dims` of the input and output should be specified."
                        )
            return swap(phi_op.conj(), dim=dims)
    raise ValueError(
        "Invalid: The variable `phi_op` must either be a list of "
        "Kraus operators or as a Choi matrix."
    )
