"""Operations on quantum channels."""
from typing import List, Union

import numpy as np

from toqito.matrix_ops import vec
from toqito.perms import permute_systems, swap
from toqito.states import max_entangled


__all__ = ["apply_map", "kraus_to_choi", "partial_map"]


def apply_map(
    mat: np.ndarray, phi_op: Union[np.ndarray, List[List[np.ndarray]]]
) -> np.ndarray:
    r"""
    Apply a superoperator to an operator.

    Examples
    ==========

    The swap operator is the Choi matrix of the transpose map. The following is
    a (non-ideal, but illustrative) way of computing the transpose of a matrix.

    Consider the following matrix

    .. math::
        X = \begin{pmatrix}
                1 & 4 & 7 \\
                2 & 5 & 8 \\
                3 & 6 & 9
            \end{pmatrix}

    Applying the swap operator given as

    .. math::
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
        X_{swap} = \begin{pmatrix}
                        1 & 2 & 3 \\
                        4 & 5 & 6 \\
                        7 & 8 & 9
                   \end{pmatrix}

    Using `toqito`, we can obtain the above matrices as follows.

    >>> from toqito.channel_ops import apply_map
    >>> from toqito.perms import swap_operator
    >>> import numpy as np
    >>> test_input_mat = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    >>> apply_map(test_input_mat, swap_operator(3))
    [[1., 2., 3.],
     [4., 5., 6.],
     [7., 8., 9.]]

    :param mat: A matrix.
    :param phi_op: A superoperator.
    :return: The result of applying the superoperator `phi_op` to the operator
             `mat`.

    `phi_op` should be provided either as a Choi matrix, or as a list of numpy
    arrays with either 1 or 2 columns whose entries are its Kraus operators.
    """
    # Both of the following methods of applying the superoperator are much
    # faster than naively looping through the Kraus operators or constructing
    # eigenvectors of a Choi matrix.

    # The superoperator was given as a list of Kraus operators:
    if isinstance(phi_op, list):
        s_phi_op = [len(phi_op), len(phi_op[0])]

        # Map is completely positive.
        if s_phi_op[1] == 1 or (s_phi_op[0] == 1 and s_phi_op[1] > 2):
            for i in range(s_phi_op[0]):
                phi_op[i][1] = phi_op[i][0].conj().T
        else:
            for i in range(s_phi_op[0]):
                phi_op[i][1] = phi_op[i][1].conj().T
        phi_0_list = []
        phi_1_list = []
        for i in range(s_phi_op[0]):
            phi_0_list.append(phi_op[i][0])
            phi_1_list.append(phi_op[i][1])

        k_1 = np.concatenate(phi_0_list, axis=1)
        k_2 = np.concatenate(phi_1_list, axis=0)

        a_mat = np.kron(np.identity(len(phi_op)), mat)
        return np.matmul(np.matmul(k_1, a_mat), k_2)

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
        )
        return np.matmul(a_mat, b_mat)
    raise ValueError(
        "Invalid: The variable `phi_op` must either be a list of "
        "Kraus operators or as a Choi matrix."
    )


def kraus_to_choi(kraus_ops: List[List[np.ndarray]], sys: int = 2) -> np.ndarray:
    r"""
    Compute the Choi matrix of a list of Kraus operators [WatKraus18]_.

    The Choi matrix of the list of Kraus operators, `kraus_ops`. The default
    convention is that the Choi matrix is the result of applying the map to the
    second subsystem of the standard maximally entangled (unnormalized) state.
    The Kraus operators are expected to be input as a list of numpy arrays.

    This function was adapted from the QETLAB package.

    Examples
    ==========

    The transpose map

    The Choi matrix of the transpose map is the swap operator.

    >>> import numpy as np
    >>> from toqito.channel_ops import kraus_to_choi
    >>> kraus_1 = np.array([[1, 0], [0, 0]])
    >>> kraus_2 = np.array([[1, 0], [0, 0]]).conj().T
    >>> kraus_3 = np.array([[0, 1], [0, 0]])
    >>> kraus_4 = np.array([[0, 1], [0, 0]]).conj().T
    >>> kraus_5 = np.array([[0, 0], [1, 0]])
    >>> kraus_6 = np.array([[0, 0], [1, 0]]).conj().T
    >>> kraus_7 = np.array([[0, 0], [0, 1]])
    >>> kraus_8 = np.array([[0, 0], [0, 1]]).conj().T
    >>>
    >>> kraus_ops = [
    >>>     [kraus_1, kraus_2],
    >>>     [kraus_3, kraus_4],
    >>>     [kraus_5, kraus_6],
    >>>     [kraus_7, kraus_8],
    >>> ]
    >>> kraus_to_choi(kraus_ops)
    [[1. 0. 0. 0.]
     [0. 0. 1. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]]

    References
    ==========
    .. [WatKraus18] Watrous, John.
        "The theory of quantum information."
        Section: "Kraus representations".
        Cambridge University Press, 2018.

    :param kraus_ops: A list of Kraus operators.
    :param sys: The dimension of the system (default is 2).
    :return: The corresponding Choi matrix of the provided Kraus operators.
    """
    dim_op_1 = kraus_ops[0][0].shape[0]
    dim_op_2 = kraus_ops[0][0].shape[1]
    choi_mat = partial_map(
        max_entangled(dim_op_1, False, False)
        * max_entangled(dim_op_2, False, False).conj().T,
        kraus_ops,
        sys,
        np.array([[dim_op_1, dim_op_1], [dim_op_2, dim_op_2]]),
    )

    return choi_mat


def partial_map(
    rho: np.ndarray,
    phi_map: Union[np.ndarray, List[List[np.ndarray]]],
    sys: int = 2,
    dim: Union[List[int], np.ndarray] = None,
) -> np.ndarray:
    r"""Apply map to a subsystem of an operator [WatPMap18]_.

    Applies the operator

    .. math::
        \left(\mathbb{I} \otimes \Phi \right) \left(\rho \right).

    In other words, it is the result of applying the channel :math:`\Phi` to the
    second subsystem of :math:`\rho`, which is assumed to act on two
    subsystems of equal dimension.

    The input `phi_map` should be provided as a Choi matrix.

    Examples
    ==========

    >>> from toqito.channel_ops import partial_map
    >>> from toqito.channels import depolarizing
    >>> rho = np.array([[0.3101, -0.0220-0.0219*1j, -0.0671-0.0030*1j, -0.0170-0.0694*1j],
    >>>                 [-0.0220+0.0219*1j, 0.1008, -0.0775+0.0492*1j, -0.0613+0.0529*1j],
    >>>                 [-0.0671+0.0030*1j, -0.0775-0.0492*1j, 0.1361, 0.0602 + 0.0062*1j],
    >>>                 [-0.0170+0.0694*1j, -0.0613-0.0529*1j, 0.0602-0.0062*1j, 0.4530]])
    >>> phi_x = partial_map(rho, depolarizing(2))
    [[ 0.20545+0.j       0.     +0.j      -0.0642 +0.02495j  0.     +0.j     ]
     [ 0.     +0.j       0.20545+0.j       0.     +0.j      -0.0642 +0.02495j]
     [-0.0642 -0.02495j  0.     +0.j       0.29455+0.j       0.     +0.j     ]
     [ 0.     +0.j      -0.0642 -0.02495j  0.     +0.j       0.29455+0.j     ]]

    >>> from toqito.channel_ops import partial_map
    >>> from toqito.channels import depolarizing
    >>> rho = np.array([[0.3101, -0.0220-0.0219*1j, -0.0671-0.0030*1j, -0.0170-0.0694*1j],
    >>>                 [-0.0220+0.0219*1j, 0.1008, -0.0775+0.0492*1j, -0.0613+0.0529*1j],
    >>>                 [-0.0671+0.0030*1j, -0.0775-0.0492*1j, 0.1361, 0.0602 + 0.0062*1j],
    >>>                 [-0.0170+0.0694*1j, -0.0613-0.0529*1j, 0.0602-0.0062*1j, 0.4530]])
    >>> phi_x = partial_map(rho, depolarizing(2), 1)
    [[0.2231+0.j      0.0191-0.00785j 0.    +0.j      0.    +0.j     ]
     [0.0191+0.00785j 0.2769+0.j      0.    +0.j      0.    +0.j     ]
     [0.    +0.j      0.    +0.j      0.2231+0.j      0.0191-0.00785j]
     [0.    +0.j      0.    +0.j      0.0191+0.00785j 0.2769+0.j     ]]

    References
    ==========
    .. [WatPMap18] Watrous, John.
        The theory of quantum information.
        Cambridge University Press, 2018.

    :param rho: A matrix.
    :param phi_map: The map to partially apply.
    :param sys: Scalar or vector specifying the size of the subsystems.
    :param dim: Dimension of the subsystems. If `None`, all dimensions
                are assumed to be equal.
    :return: The partial map `phi_map` applied to matrix `rho`.
    """
    if dim is None:
        dim = np.round(np.sqrt(list(rho.shape))).conj().T * np.ones(2)
    if isinstance(dim, list):
        dim = np.array(dim)

    # Force dim to be a row vector.
    if dim.ndim == 1:
        dim = dim.T.flatten()
        dim = np.array([dim, dim])

    prod_dim_r1 = int(np.prod(dim[0, : sys - 1]))
    prod_dim_c1 = int(np.prod(dim[1, : sys - 1]))
    prod_dim_r2 = int(np.prod(dim[0, sys:]))
    prod_dim_c2 = int(np.prod(dim[1, sys:]))

    # Note: In the case where the Kraus operators refer to a CP map, this
    # approach of appending to the list may not work.
    if isinstance(phi_map, list):
        # The `phi_map` variable is provided as a list of Kraus operators.
        phi = []
        for i, _ in enumerate(phi_map):
            phi.append(
                np.kron(
                    np.kron(np.identity(prod_dim_r1), phi_map[i]),
                    np.identity(prod_dim_r2),
                )
            )
        phi_x = apply_map(rho, phi)
        return phi_x

    # The `phi_map` variable is provided as a Choi matrix.
    if isinstance(phi_map, np.ndarray):
        dim_phi = phi_map.shape

        dim = np.array(
            [
                [
                    prod_dim_r2,
                    prod_dim_r2,
                    int(dim[0, sys - 1]),
                    int(dim_phi[0] / dim[0, sys - 1]),
                    prod_dim_r1,
                    prod_dim_r1,
                ],
                [
                    prod_dim_c2,
                    prod_dim_c2,
                    int(dim[1, sys - 1]),
                    int(dim_phi[1] / dim[1, sys - 1]),
                    prod_dim_c1,
                    prod_dim_c1,
                ],
            ]
        )
        psi_r1 = max_entangled(prod_dim_r2, False, False)
        psi_c1 = max_entangled(prod_dim_c2, False, False)
        psi_r2 = max_entangled(prod_dim_r1, False, False)
        psi_c2 = max_entangled(prod_dim_c1, False, False)

        phi_map = permute_systems(
            np.kron(
                np.kron(psi_r1 * psi_c1.conj().T, phi_map), psi_r2 * psi_c2.conj().T
            ),
            [1, 3, 5, 2, 4, 6],
            dim,
        )

        phi_x = apply_map(rho, phi_map)

        return phi_x

    raise ValueError(
        "The `phi_map` variable is assumed to be provided as "
        "either a Choi matrix or a list of Kraus operators."
    )
