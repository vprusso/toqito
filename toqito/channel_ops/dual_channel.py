"""Computes the dual of a map."""

from typing import Optional, Union, cast

import numpy as np

from toqito.helper import channel_dim
from toqito.perms import swap


def dual_channel(
    phi_op: Union[np.ndarray, list[np.ndarray], list[list[np.ndarray]]], dims: Optional[list[int]] = None
) -> Union[np.ndarray, list[list[np.ndarray]], list[np.ndarray]]:
    r"""Compute the dual of a map (quantum channel).

    (Section: Representations and Characterizations of Channels of :cite:`Watrous_2018_TQI`).

    The map can be represented as a Choi matrix, with optional specification of input
    and output dimensions. If the input channel maps :math:`M_{r,c}` to :math:`M_{x,y}`
    then :code:`dim` should be the list :code:`[[r,x], [c,y]]`. If it maps :math:`M_m`
    to :math:`M_n`, then :code:`dim` can simply be the vector :code:`[m,n]`. In this
    case the Choi matrix of the dual channel is returned, obtained by swapping input and
    output (see :func:`.swap`), and complex conjugating all elements.

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

    Examples
    ========
    When a channel is represented by a 1-D list of of Kraus operators, the CPTP dual channel can be determined
    as shown below.

    >>> import numpy as np
    >>> from toqito.channel_ops import dual_channel
    >>> kraus_1 = np.array([[1, 0, 1j, 0]])
    >>> kraus_2 = np.array([[0, 1, 0, 1j]])
    >>> kraus_list = [kraus_1, kraus_2]
    >>> dual_channel(kraus_list)
    [array([[1.-0.j],
           [0.-0.j],
           [0.-1.j],
           [0.-0.j]]), array([[0.-0.j],
           [1.-0.j],
           [0.-0.j],
           [0.-1.j]])]

    If the input channel's dimensions are different from the output dual channel's dimensions,

    >>> import numpy as np
    >>> from toqito.channel_ops import dual_channel
    >>> from toqito.perms import swap_operator
    >>> input_op = swap_operator([2, 3])
    >>> dual_channel(input_op, [[3, 2], [2, 3]])
    array([[1., 0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0.],
           [0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 1.]])

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If matrices are not Choi matrix.
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
            phi_op_2d = cast(list[list[np.ndarray]], phi_op)
            return [[a.conj().T for a in inner] for inner in phi_op_2d]
        if isinstance(phi_op[0], np.ndarray):
            return [a.conj().T for a in phi_op if isinstance(a, np.ndarray)]

    # If phi_op is a `ndarray`, assume it is a Choi matrix.
    if isinstance(phi_op, np.ndarray):
        if len(phi_op.shape) == 2:
            if dims is None:
                raise ValueError("When using a Choi Matrix, dims must be provided")
            dims_result = channel_dim(phi_op, dim=dims, compute_env_dim=False)
            if not isinstance(dims_result, list) or len(dims_result) != 2:
                raise TypeError("Expected `channel_dim` to return a list of two items.")
            d_in, d_out, _ = dims_result
            return swap(phi_op.conj(), dim=[[d_in[0], d_out[0]], [d_in[1], d_out[1]]])
    raise ValueError("Invalid: The variable `phi_op` must either be a list of Kraus operators or as a Choi matrix.")
