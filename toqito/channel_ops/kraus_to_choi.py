"""Computes the Choi matrix of a list of Kraus operators."""

import numpy as np

from toqito.channel_ops import partial_channel
from toqito.helper import channel_dim
from toqito.states import max_entangled


def kraus_to_choi(kraus_ops: list[list[np.ndarray]], sys: int = 2) -> np.ndarray:
    r"""Compute the Choi matrix of a list of Kraus operators.

    (Section: Kraus Representations of :footcite:`Watrous_2018_TQI`).

    The Choi matrix of the list of Kraus operators, :code:`kraus_ops`. The default convention is
    that the Choi matrix is the result of applying the map to the second subsystem of the
    standard maximally entangled (unnormalized) state. The Kraus operators are expected to be
    input as a list of numpy arrays (i.e. [[:code:`A_1`, :code:`B_1`],...,[:code:`A_n`, :code:`B_n`]]).
    In case the map is CP (completely positive), it suffices to input a flat list of operators omitting
    their conjugate transpose (i.e. [:math:`K_1`,..., :math:`K_n`]).

    This function was adapted from the QETLAB package.

    Examples
    ==========

    The transpose map:

    The Choi matrix of the transpose map is the swap operator. Notice that the transpose map
    is *not* completely positive.

    .. jupyter-execute::

     import numpy as np
     from toqito.channel_ops import kraus_to_choi
     kraus_1 = np.array([[1, 0], [0, 0]])
     kraus_2 = np.array([[1, 0], [0, 0]]).conj().T
     kraus_3 = np.array([[0, 1], [0, 0]])
     kraus_4 = np.array([[0, 1], [0, 0]]).conj().T
     kraus_5 = np.array([[0, 0], [1, 0]])
     kraus_6 = np.array([[0, 0], [1, 0]]).conj().T
     kraus_7 = np.array([[0, 0], [0, 1]])
     kraus_8 = np.array([[0, 0], [0, 1]]).conj().T

     kraus_ops = [[kraus_1, kraus_2], [kraus_3, kraus_4], [kraus_5, kraus_6], [kraus_7, kraus_8]]
     choi_op = kraus_to_choi(kraus_ops)
     print(choi_op)

    See Also
    ========
    :py:func:`~toqito.channel_ops.choi_to_kraus.choi_to_kraus`

    References
    ==========
    .. footbibliography::



    :param kraus_ops: A list of Kraus operators.
    :param sys: The dimension of the system (default is 2).
    :return: The corresponding Choi matrix of the provided Kraus operators.

    """
    dim_in, _, _ = channel_dim(kraus_ops)
    dim_op_1, dim_op_2 = dim_in

    choi_mat = partial_channel(
        max_entangled(dim_op_1, False, False) @ max_entangled(dim_op_2, False, False).conj().T,
        kraus_ops,
        sys,
        np.array([[dim_op_1, dim_op_1], [dim_op_2, dim_op_2]]),
    )

    return choi_mat
