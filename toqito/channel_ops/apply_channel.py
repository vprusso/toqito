"""Apply channel to an operator."""


import itertools

import numpy as np

from toqito.matrix_ops import vec
from toqito.perms import swap


def apply_channel(mat: np.ndarray, phi_op: np.ndarray | list[list[np.ndarray]]) -> np.ndarray:
    r"""Apply a quantum channel to an operator.

    (Section: Representations and Characterizations of Channels of :cite:`Watrous_2018_TQI`).

    Specifically, an application of the channel is defined as

    .. math::
        \Phi(X) = \text{Tr}_{\mathcal{X}} \left(J(\Phi)
        \left(\mathbb{I}_{\mathcal{Y}} \otimes X^{T}\right)\right),

    where

    .. math::
        J(\Phi): \text{T}(\mathcal{X}, \mathcal{Y}) \rightarrow
        \text{L}(\mathcal{Y} \otimes \mathcal{X})

    is the Choi representation of :math:`\Phi`.

    We assume the quantum channel given as :code:`phi_op` is provided as either the Choi matrix
    of the channel or a set of Kraus operators that define the quantum channel.

    This function is adapted from the QETLAB package.

    Examples
    ==========

    The swap operator is the Choi matrix of the transpose map. The following is a (non-ideal,
    but illustrative) way of computing the transpose of a matrix.

    Consider the following matrix

    .. math::
        X = \begin{pmatrix}
                1 & 4 & 7 \\
                2 & 5 & 8 \\
                3 & 6 & 9
            \end{pmatrix}

    Applying the swap operator given as

    .. math::
        \Phi =
        \begin{pmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
         \end{pmatrix}

    to the matrix :math:`X`, we have the resulting matrix of

    .. math::
        \Phi(X) = \begin{pmatrix}
                        1 & 2 & 3 \\
                        4 & 5 & 6 \\
                        7 & 8 & 9
                   \end{pmatrix}

    Using :code:`toqito`, we can obtain the above matrices as follows.

    >>> from toqito.channel_ops import apply_channel
    >>> from toqito.perms import swap_operator
    >>> import numpy as np
    >>> test_input_mat = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    >>> apply_channel(test_input_mat, swap_operator(3))
    [[1., 2., 3.],
     [4., 5., 6.],
     [7., 8., 9.]]

    References
    ==========
    .. bibliography::
        :filter: docname in docnames


    :raises ValueError: If matrix is not Choi matrix.
    :param mat: A matrix.
    :param phi_op: A superoperator. :code:`phi_op` should be provided either as a Choi matrix,
                   or as a list of numpy arrays with either 1 or 2 columns whose entries are its
                   Kraus operators.
    :return: The result of applying the superoperator :code:`phi_op` to the operator :code:`mat`.

    """
    # Both of the following methods of applying the superoperator are much faster than naively
    # looping through the Kraus operators or constructing eigenvectors of a Choi matrix.

    # The superoperator was given as a list of Kraus operators:
    if isinstance(phi_op, list):
        s_phi_op = [len(phi_op), len(phi_op[0])]

        phi_0_list = []
        phi_1_list = []

        # Map is completely positive if input is given as:
        # 1. [K1, K2, .. Kr]
        # 2. [[K1], [K2], .. [Kr]]
        # 3. [[K1, K2, .. Kr]] and r > 2
        if isinstance(phi_op[0], np.ndarray):
            phi_0_list = phi_op
        elif s_phi_op[1] == 1 or (s_phi_op[0] == 1 and s_phi_op[1] > 2):
            phi_0_list = list(itertools.chain(*phi_op))
        else:
            # Input is given as: [[A1, B1], [A2, B2], .. [Ar, Br]]
            phi_0_list = [k_mat[0] for k_mat in phi_op]
            phi_1_list = [k_mat[1].conj().T for k_mat in phi_op]

        if not phi_1_list:
            phi_1_list = [k_mat.conj().T for k_mat in phi_0_list]

        k_1 = np.concatenate(phi_0_list, axis=1)
        k_2 = np.concatenate(phi_1_list, axis=0)

        a_mat = np.kron(np.identity(len(phi_0_list)), mat)
        return k_1 @ a_mat @ k_2

    # The superoperator was given as a Choi matrix:
    if isinstance(phi_op, np.ndarray):
        mat_size = np.array(list(mat.shape))
        phi_size = np.array(list(phi_op.shape)) / mat_size

        a_mat = np.kron(vec(mat).T[0], np.identity(int(phi_size[0])))
        b_mat = np.reshape(
            swap(
                phi_op.T,
                [1, 2],
                [[mat_size[1], phi_size[1]], [mat_size[0], phi_size[0]]],
                True,
            ).T,
            (int(phi_size[0] * np.prod(mat_size)), int(phi_size[1])),
            order="F",
        )
        return a_mat @ b_mat
    raise ValueError(
        "Invalid: The variable `phi_op` must either be a list of "
        "Kraus operators or as a Choi matrix."
    )
