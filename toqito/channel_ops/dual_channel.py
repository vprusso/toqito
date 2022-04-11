"""Compute the dual of a map."""
from typing import Union
import numpy as np

from toqito.matrix_props import is_square
from toqito.perms import swap


def dual_channel(
    phi_op: Union[np.ndarray, list[np.ndarray], list[list[np.ndarray]]], dims: list[int] = None
) -> Union[np.ndarray, list[list[np.ndarray]]]:
    r"""
    Compute the dual of a map (quantum channel) [WatDChan18]_.

    The map can be represented as a Choi matrix, with optional specification of input
    and output dimensions. In this case the Choi matrix of the dual channel is
    returned, obtained by swapping input and output (see :func:`toqito.perms.swap`),
    and complex conjugating all elements.

    The map can also be represented as a list of Kraus operators.
    A list of lists, each containing two elements, corresponds to the families
    of operators :math:`\{(A_a, B_a)\}` representing the map

    .. math::
        \Phi(X) = \sum_a A_a X B^*_a.

    The dual map is obtained by taking the Hermitian adjoint of each operator.
    If :code:`phi_op` is given as a one-dimensional list, :math:`\{A_a\}`,
    it is interpreted as the completely positive map

    .. math::
        \Phi(X) = \sum_a A_a X A^*_a.

    References
    ==========
    .. [WatDChan18] Watrous, John.
        The theory of quantum information.
        Section: Representations and characterizations of channels.
        Cambridge University Press, 2018.

    :param phi_op: A superoperator. It should be provided either as a Choi matrix,
                   or as a (1d or 2d) list of numpy arrays whose entries are its Kraus operators.
    :param dims: Dimension of the input and output systems, for Choi matrix representation.
                 If :code:`None`, try to infer them from :code:`phi_op.shape`.
    :return: The map dual to :code:`phi_op`, in the same representation.
    """
    # If phi_op is a list, assume it contains couples of Kraus operators
    # and take the Hermitian conjugate.
    if isinstance(phi_op, list):
        if isinstance(phi_op[0], list):
            return [[a.conj().T for a in x] for x in phi_op]
        if isinstance(phi_op[0], np.ndarray):
            return [a.conj().T for a in phi_op]

    # If phi_op is a `ndarray`, assume it is a Choi matrix.
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
