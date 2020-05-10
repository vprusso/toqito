"""Quantum channels."""
from typing import List, Union

import functools
import operator
import numpy as np

from scipy.sparse import identity

from toqito.matrix_ops import vec
from toqito.perms import permute_systems, swap
from toqito.states import max_entangled
from toqito.helper import expr_as_np_array, np_array_as_expr


__all__ = [
    "apply_map",
    "choi_map",
    "dephasing",
    "depolarizing",
    "partial_map",
    "partial_trace",
    "partial_trace_cvx",
    "partial_transpose",
    "realignment",
    "reduction_map",
]


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


def choi_map(a_var: int = 1, b_var: int = 1, c_var: int = 0) -> np.ndarray:
    r"""
    Produce the Choi map or one of its generalizations [Choi92]_.

    The *Choi map* is a positive map on 3-by-3 matrices that is capable
    of detecting some entanglement that the transpose map is not.

    The standard Choi map defined with `a=1`, `b=1`, and `c=0` is the
    Choi matrix of the positive map defined in [Choi92]_. Many of these
    maps are capable of detecting PPT entanglement.

    Examples
    ==========

    The standard Choi map is given as

    .. math::
        \begin{pmatrix}
            1 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & -1 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            -1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & -1 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            -1 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & 1
        \end{pmatrix}

    We can generate the Choi map in `toqito` as follows.

    >>> from toqito.channels import choi_map
    >>> import numpy as np
    >>> choi_map()
    [[ 1.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.],
     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
     [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0., -1.],
     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
     [-1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  1.]])

    The reduction map is the map :math:`R` defined by:

    .. math::
        R(X) = \text{Tr}(X) \mathbb{I} - X.

    The matrix correspond to this is given as

    .. math::
        \begin{pmatrix}
            0 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & -1 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            -1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -1 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            -1 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & 0
        \end{pmatrix}

    The reduction map is the Choi map that arises when :math:`a = 0` and when
    :math:`b = c = 1`. We can obtain this matrix using `toqito` as follows.

    >>> from toqito.channels import choi_map
    >>> import numpy as np
    >>> choi_map(0, 1, 1)
    [[ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.],
     [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
     [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
     [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
     [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
     [-1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.]])

    References
    ==========
    .. [Choi92] Cho, Sung Je, Seung-Hyeok Kye, and Sa Ge Lee.
        "Generalized Choi maps in three-dimensional matrix algebra."
        Linear algebra and its applications 171 (1992): 213-224.
        https://www.sciencedirect.com/science/article/pii/002437959290260H

    :param a_var: Default integer for standard Choi map.
    :param b_var: Default integer for standard Choi map.
    :param c_var: Default integer for standard Choi map.
    :return: The Choi map (or one of its  generalizations).
    """
    psi = max_entangled(3, False, False)
    return (
        np.diag(
            [a_var + 1, c_var, b_var, b_var, a_var + 1, c_var, c_var, b_var, a_var + 1]
        )
        - psi * psi.conj().T
    )


def dephasing(dim: int, param_p: float = 0) -> np.ndarray:
    r"""
    Produce the partially dephasing channel [WatDeph18]_.

    The Choi matrix of the completely dephasing channel that acts on
    `dim`-by-`dim` matrices.

    Let :math:`\Sigma` be an alphabet and let
    :math:`\mathcal{X} = \mathbb{C}^{\Sigma}`. The map
    :math:`\Delta \in \text{T}(\mathcal{X})` defined as

    .. math::
        \Delta(X) = \sum_{a \in \Sigma} X(a, a) E_{a,a}

    for every :math:`X \in \text{L}(\mathcal{X})` is defined as the *completely
    dephasing channel*.

    Examples
    ==========

    The completely dephasing channel maps kills everything off the diagonal.
    Consider the following matrix

    .. math::
        \rho = \begin{pmatrix}
                   1 & 2 & 3 & 4 \\
                   5 & 6 & 7 & 8 \\
                   9 & 10 & 11 & 12 \\
                   13 & 14 & 15 & 16
               \end{pmatrix}.

    Applying the dephasing channel to :math:`\rho` we have that

    .. math::
        \Phi(\rho) = \begin{pmatrix}
                         1 & 0 & 0 & 0 \\
                         0 & 6 & 0 & 0 \\
                         0 & 0 & 11 & 0 \\
                         0 & 0 & 0 & 16
                     \end{pmatrix}.

    This can be observed in `toqito` as follows.

    >>> from toqito.channel_ops import apply_map
    >>> from toqito.channels import dephasing
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> apply_map(test_input_mat, dephasing(4))
    [[ 1.,  0.,  0.,  0.],
     [ 0.,  6.,  0.,  0.],
     [ 0.,  0., 11.,  0.],
     [ 0.,  0.,  0., 16.]])

    We may also consider setting the parameter `p = 0.5`.

    >>> from toqito.channel_ops import apply_map
    >>> from toqito.channels import dephasing
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> apply_map(test_input_mat, dephasing(4, 0.5))
    [[17.5  0.   0.   0. ]
     [ 0.  20.   0.   0. ]
     [ 0.   0.  22.5  0. ]
     [ 0.   0.   0.  25. ]]

    References
    ==========
    .. [WatDeph18] Watrous, John.
        "The theory of quantum information."
        Section: "The completely dephasing channel".
        Cambridge University Press, 2018.

    :param dim: The dimensionality on which the channel acts.
    :param param_p: Default is 0.
    :return: The Choi matrix of the dephasing channel.
    """
    # Compute the Choi matrix of the dephasing channel.

    # Gives a sparse non-normalized state.
    psi = max_entangled(dim=dim, is_sparse=False, is_normalized=False)
    return (1 - param_p) * np.diag(np.diag(psi * psi.conj().T)) + param_p * (
        psi * psi.conj().T
    )


def depolarizing(dim: int, param_p: float = 0) -> np.ndarray:
    r"""
    Produce the partially depolarizng channel [WikDepo]_, [WatDepo18]_.

    The Choi matrix of the completely depolarizing channel that acts on
    `dim`-by-`dim` matrices.

    The *completely depolarizing channel* is defined as

    .. math::
        \Omega(X) = \text{Tr}(X) \omega

    for all :math:`X \in \text{L}(\mathcal{X})`, where

    .. math::
        \omega = \frac{\mathbb{I}_{\mathcal{X}}}{\text{dim}(\mathcal{X})}

    denotes the completely mixed stated defined with respect to the space
    :math:`\mathcal{X}`.

    Examples
    ==========

    The completely depolarizing channel maps every density matrix to the
    maximally-mixed state. For example, consider the density operator

    .. math::
        \rho = \frac{1}{2} \begin{pmatrix}
                             1 & 0 & 0 & 1 \\
                             0 & 0 & 0 & 0 \\
                             0 & 0 & 0 & 0 \\
                             1 & 0 & 0 & 1
                           \end{pmatrix}

    corresponding to one of the Bell states. Applying the depolarizing channel
    to :math:`\rho` we have that

    .. math::
        \Phi(\rho) = \frac{1}{4} \begin{pmatrix}
                                    \frac{1}{2} & 0 & 0 & \frac{1}{2} \\
                                    0 & 0 & 0 & 0 \\
                                    0 & 0 & 0 & 0 \\
                                    \frac{1}{2} & 0 & 0 & \frac{1}{2}
                                 \end{pmatrix}.

    This can be observed in `toqito` as follows.

    >>> from toqito.channel_ops import apply_map
    >>> from toqito.channels import depolarizing
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1 / 2, 0, 0, 1 / 2], [0, 0, 0, 0], [0, 0, 0, 0], [1 / 2, 0, 0, 1 / 2]]
    >>> )
    >>> apply_map(test_input_mat, depolarizing(4))
    [[0.125 0.    0.    0.125]
     [0.    0.    0.    0.   ]
     [0.    0.    0.    0.   ]
     [0.125 0.    0.    0.125]]

    >>> from toqito.channel_ops import apply_map
    >>> from toqito.channels import depolarizing
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> apply_map(test_input_mat, depolarizing(4, 0.5))
    [[17.125  0.25   0.375  0.5  ]
     [ 0.625 17.75   0.875  1.   ]
     [ 1.125  1.25  18.375  1.5  ]
     [ 1.625  1.75   1.875 19.   ]]


    References
    ==========
    .. [WikDepo] Wikipedia: Quantum depolarizing channel
        https://en.wikipedia.org/wiki/Quantum_depolarizing

    .. [WatDepo18] Watrous, John.
        "The theory of quantum information."
        Section: "Replacement channels and the completely depolarizing channel".
        Cambridge University Press, 2018.

    :param dim: The dimensionality on which the channel acts.
    :param param_p: Default 0.
    :return: The Choi matrix of the completely depolarizing channel.
    """
    # Compute the Choi matrix of the depolarizng channel.

    # Gives a sparse non-normalized state.
    psi = max_entangled(dim=dim, is_sparse=False, is_normalized=False)
    return (1 - param_p) * identity(dim ** 2) / dim + param_p * (psi * psi.conj().T)


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

    >>> from toqito.channels import partial_map
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

    >>> from toqito.channels import partial_map
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


def partial_trace_cvx(rho, sys=None, dim=None):
    """
    Perform the partial trace on a cvxpy variable.

    Adapted from [CVXPtrace]_.

    References
    ==========
    .. [CVXPtrace] Partial trace for CVXPY variables
        https://github.com/cvxgrp/cvxpy/issues/563

    :param rho: A square matrix.
    :param sys: Scalar or vector specifying the size of the subsystems.
    :param dim: Dimension of the subsystems. If `None`, all dimensions
                are assumed to be equal.
    :return: The partial trace of matrix `input_mat`.
    """
    rho_np = expr_as_np_array(rho)
    traced_rho = partial_trace(rho_np, sys, dim)
    traced_rho = np_array_as_expr(traced_rho)
    return traced_rho


def partial_trace(
    input_mat: np.ndarray,
    sys: Union[int, List[int]] = 2,
    dim: Union[int, List[int]] = None,
):
    r"""
    Compute the partial trace of a matrix [WikPtrace]_.

    Gives the partial trace of the matrix X, where the dimensions of the
    (possibly more than 2) subsystems are given by the vector `dim` and the
    subsystems to take the trace on are given by the scalar or vector `sys`.

    Examples
    ==========

    Consider the following matrix

    .. math::
        X = \begin{pmatrix}
                1 & 2 & 3 & 4 \\
                5 & 6 & 7 & 8 \\
                9 & 10 & 11 & 12 \\
                13 & 14 & 15 & 16
            \end{pmatrix}.

    Taking the partial trace over the second subsystem of :math:`X` yields the
    following matrix

    .. math::
        X_{pt, 2} = \begin{pmatrix}
                    7 & 11 \\
                    23 & 27
                 \end{pmatrix}

    By default, the partial trace function in `toqito` takes the trace of the
    second subsystem.

    >>> from toqito.channels import partial_trace
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> partial_trace(test_input_mat)
    [[ 7, 11],
     [23, 27]]

    By specifying the `sys = 1` argument, we can perform the partial trace over
    the first subsystem (instead of the default second subsystem as done above).
    Performing the partial trace over the first subsystem yields the following
    matrix

    .. math::
        X_{pt, 1} = \begin{pmatrix}
                        12 & 14 \\
                        20 & 22
                    \end{pmatrix}

    >>> from toqito.channels import partial_trace
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> partial_trace(test_input_mat, 1)
    [[12, 14],
     [20, 22]]

    We can also specify both dimension and system size as `list` arguments.
    Consider the following :math:`16`-by-:math:`16` matrix.

    >>> from toqito.channels import partial_trace
    >>> import numpy as np
    >>> test_input_mat = np.arange(1, 257).reshape(16, 16)
    >>> test_input_mat
    [[  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16]
     [ 17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32]
     [ 33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48]
     [ 49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64]
     [ 65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80]
     [ 81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96]
     [ 97  98  99 100 101 102 103 104 105 106 107 108 109 110 111 112]
     [113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128]
     [129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144]
     [145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160]
     [161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176]
     [177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192]
     [193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208]
     [209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224]
     [225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240]
     [241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256]]

    We can take the partial trace on the first and third subsystems and assume
    that the size of each of the 4 systems is of dimension 2.

    >>> from toqito.channels import partial_trace
    >>> import numpy as np
    >>> partial_trace(test_input_mat, [1, 3], [2, 2, 2, 2])
    [[344, 348, 360, 364],
     [408, 412, 424, 428],
     [600, 604, 616, 620],
     [664, 668, 680, 684]])

    References
    ==========
    .. [WikPtrace] Wikipedia: Partial trace
        https://en.wikipedia.org/wiki/Partial_trace

    :param input_mat: A square matrix.
    :param sys: Scalar or vector specifying the size of the subsystems.
    :param dim: Dimension of the subsystems. If `None`, all dimensions
                are assumed to be equal.
    :return: The partial trace of matrix `input_mat`.
    """
    if dim is None:
        dim = np.array([np.round(np.sqrt(len(input_mat)))])
    if isinstance(dim, int):
        dim = np.array([dim])
    if isinstance(dim, list):
        dim = np.array(dim)

    if sys is None:
        sys = 2

    num_sys = len(dim)

    # Allow the user to enter a single number for dim.
    if num_sys == 1:
        dim = np.array([dim[0], len(input_mat) / dim[0]])
        if (
            np.abs(dim[1] - np.round(dim[1]))
            >= 2 * len(input_mat) * np.finfo(float).eps
        ):
            raise ValueError(
                "Invalid: If `dim` is a scalar, `dim` must evenly "
                "divide `len(input_mat)`."
            )
        dim[1] = np.round(dim[1])
        num_sys = 2

    prod_dim = np.prod(dim)
    if isinstance(sys, list):
        prod_dim_sys = np.prod(dim[sys])
    elif isinstance(sys, int):
        prod_dim_sys = np.prod(dim[sys - 1])
    else:
        raise ValueError(
            "Invalid: The variable `sys` must either be of type "
            "int or of a list of ints."
        )

    sub_prod = prod_dim / prod_dim_sys
    sub_sys_vec = prod_dim * np.ones(int(sub_prod)) / sub_prod

    if isinstance(sys, int):
        sys = [sys]
    set_diff = list(set(list(range(1, num_sys + 1))) - set(sys))

    perm = set_diff
    perm.extend(sys)

    a_mat = permute_systems(input_mat, perm, dim)

    ret_mat = np.reshape(
        a_mat,
        [int(sub_sys_vec[0]), int(sub_prod), int(sub_sys_vec[0]), int(sub_prod)],
        order="F",
    )
    permuted_mat = ret_mat.transpose((1, 3, 0, 2))
    permuted_reshaped_mat = np.reshape(
        permuted_mat,
        [int(sub_prod), int(sub_prod), int(sub_sys_vec[0] ** 2)],
        order="F",
    )

    pt_mat = permuted_reshaped_mat[
        :, :, list(range(0, int(sub_sys_vec[0] ** 2), int(sub_sys_vec[0] + 1)))
    ]
    pt_mat = np.sum(pt_mat, axis=2)

    return pt_mat


def partial_transpose(
    rho: np.ndarray,
    sys: Union[List[int], np.ndarray, int] = 2,
    dim: Union[List[int], np.ndarray] = None,
) -> np.ndarray:
    r"""Compute the partial transpose of a matrix [WikPtrans]_.

    By default, the returned matrix is the partial transpose of the matrix
    `rho`, where it is assumed that the number of rows and columns of `rho` are
    both perfect squares and both subsystems have equal dimension. The
    transpose is applied to the second subsystem.

    In the case where `sys` amd `dim` are specified, this function gives the
    partial transpose of the matrix `rho` where the dimensions of the (possibly
    more than 2) subsystems are given by the vector `dim` and the subsystems to
    take the partial transpose are given by the scalar or vector `sys`. If
    `rho` is non-square, different row and column dimensions can be specified
    by putting the row dimensions in the first row of `dim` and the column
    dimensions in the second row of `dim`.

    Examples
    ==========

    Consider the following matrix

    .. math::
        X = \begin{pmatrix}
                1 & 2 & 3 & 4 \\
                5 & 6 & 7 & 8 \\
                9 & 10 & 11 & 12 \\
                13 & 14 & 15 & 16
            \end{pmatrix}.

    Performing the partial transpose on the matrix :math:`X` over the second
    subsystem yields the following matrix

    .. math::
        X_{pt, 2} = \begin{pmatrix}
                    1 & 5 & 3 & 7 \\
                    2 & 6 & 4 & 8 \\
                    9 & 13 & 11 & 15 \\
                    10 & 14 & 12 & 16
                 \end{pmatrix}.

    By default, in `toqito`, the partial transpose function performs the
    transposition on the second subsystem as follows.

    >>> from toqito.channels import partial_transpose
    >>> import numpy as np
    >>> test_input_mat = np.arange(1, 17).reshape(4, 4)
    >>> partial_transpose(test_input_mat)
    [[ 1  5  3  7]
     [ 2  6  4  8]
     [ 9 13 11 15]
     [10 14 12 16]]

    By specifying the `sys = 1` argument, we can perform the partial transpose
    over the first subsystem (instead of the default second subsystem as done
    above). Performing the partial transpose over the first subsystem yields the
    following matrix

    .. math::
        X_{pt, 1} = \begin{pmatrix}
                        1 & 2 & 9 & 10 \\
                        5 & 6 & 13 & 14 \\
                        3 & 4 & 11 & 12 \\
                        7 & 8 & 15 & 16
                    \end{pmatrix}

    >>> from toqito.channels import partial_transpose
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> partial_transpose(test_input_mat, 1)
    [[ 1  2  9 10]
     [ 5  6 13 14]
     [ 3  4 11 12]
     [ 7  8 15 16]]

    References
    ==========
    .. [WikPtrans] Wikipedia: Partial transpose
        https://en.wikipedia.org/w/index.php?title=Partial_transpose&redirect=no

    :param rho: A matrix.
    :param sys: Scalar or vector specifying the size of the subsystems.
    :param dim: Dimension of the subsystems. If `None`, all dimensions
                are assumed to be equal.
    :returns: The partial transpose of matrix `rho`.
    """
    sqrt_rho_dims = np.round(np.sqrt(list(rho.shape)))

    if dim is None:
        dim = np.array(
            [[sqrt_rho_dims[0], sqrt_rho_dims[0]], [sqrt_rho_dims[1], sqrt_rho_dims[1]]]
        )
    if isinstance(dim, float):
        dim = np.array([dim])
    if isinstance(dim, list):
        dim = np.array(dim)
    if isinstance(sys, list):
        sys = np.array(sys)
    if isinstance(sys, int):
        sys = np.array([sys])

    num_sys = len(dim)
    # Allow the user to enter a single number for dim.
    if num_sys == 1:
        dim = np.array([dim, list(rho.shape)[0] / dim])
        if (
            np.abs(dim[1] - np.round(dim[1]))[0]
            >= 2 * list(rho.shape)[0] * np.finfo(float).eps
        ):
            raise ValueError(
                "InvalidDim: If `dim` is a scalar, `rho` must be "
                "square and `dim` must evenly divide `len(rho)`; "
                "please provide the `dim` array containing the "
                "dimensions of the subsystems."
            )
        dim[1] = np.round(dim[1])
        num_sys = 2

    # Allow the user to enter a vector for dim if X is square.
    if min(dim.shape) == 1 or len(dim.shape) == 1:
        # Force dim to be a row vector.
        dim = dim.T.flatten()
        dim = np.array([dim, dim])

    prod_dim_r = int(np.prod(dim[0, :]))
    prod_dim_c = int(np.prod(dim[1, :]))

    sub_prod_r = np.prod(dim[0, sys - 1])
    sub_prod_c = np.prod(dim[1, sys - 1])

    sub_sys_vec_r = prod_dim_r * np.ones(int(sub_prod_r)) / sub_prod_r
    sub_sys_vec_c = prod_dim_c * np.ones(int(sub_prod_c)) / sub_prod_c

    set_diff = list(set(list(range(1, num_sys + 1))) - set(sys))
    perm = sys.tolist()[:]
    perm.extend(set_diff)

    # Permute the subsystems so that we just have to do the partial transpose
    # on the first (potentially larger) subsystem.
    rho_permuted = permute_systems(rho, perm, dim)

    x_tmp = np.reshape(
        rho_permuted,
        [
            int(sub_sys_vec_r[0]),
            int(sub_prod_r),
            int(sub_sys_vec_c[0]),
            int(sub_prod_c),
        ],
        order="F",
    )
    y_tmp = np.transpose(x_tmp, [0, 3, 2, 1])
    z_tmp = np.reshape(y_tmp, [int(prod_dim_r), int(prod_dim_c)], order="F")

    # # Return the subsystems back to their original positions.
    # if len(sys) > 1:
    #     dim[:, sys-1] = dim[[1, 0], sys-1]

    dim = dim[:, np.array(perm) - 1]

    return permute_systems(z_tmp, perm, dim, False, True)


def realignment(input_mat: np.ndarray, dim=None) -> np.ndarray:
    r"""
    Compute the realignment of a bipartite operator [REALIGN]_.

    Gives the realignment of the matrix `input_mat`, where it is assumed that
    the number of rows and columns of `input_mat` are both perfect squares and
    both subsystems have equal dimension. The realignment is defined by mapping
    the operator :math:`|ij \rangle \langle kl |` to :math:`|ik \rangle \langle
    jl |` and extending linearly.

    If `input_mat` is non-square, different row and column dimensions can be
    specified by putting the row dimensions in the first row of `dim` and the
    column dimensions in the second row of `dim`.

    Examples
    ==========

    The standard realignment map

    Using `toqito`, we can generate the standard realignment map as follows.
    When viewed as a map on block matrices, the realignment map takes each block
    of the original matrix and makes its vectorization the rows of the
    realignment matrix. This is illustrated by the following small example:

    >>> from toqito.channels import realignment
    >>> import numpy as np
    >>> test_input_mat = np.array(
    >>>     [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> )
    >>> realignment(test_input_mat)
    [[ 1  2  5  6]
     [ 3  4  7  8]
     [ 9 10 13 14]
     [11 12 15 16]]

    References
    ==========
    .. [REALIGN] Lupo, Cosmo, Paolo Aniello, and Antonello Scardicchio.
        "Bipartite quantum systems: on the realignment criterion and beyond."
        Journal of Physics A: Mathematical and Theoretical
        41.41 (2008): 415301.
        https://arxiv.org/abs/0802.2019

    :param input_mat: The input matrix.
    :param dim: Default has all equal dimensions.
    :return: The realignment map matrix.
    """
    eps = np.finfo(float).eps
    dim_mat = input_mat.shape
    round_dim = np.round(np.sqrt(dim_mat))
    if dim is None:
        dim = np.transpose(np.array([round_dim]))
    if isinstance(dim, list):
        dim = np.array(dim)

    if isinstance(dim, int):
        dim = np.array([[dim], [dim_mat[0] / dim]])
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * dim_mat[0] * eps:
            raise ValueError("InvalidDim:")
        dim[1] = np.round(dim[1])
        dim = np.array([[1], [4]])

    if min(dim.shape) == 1:
        dim = dim[:].T
        dim = functools.reduce(operator.iconcat, dim, [])
        dim = np.array([dim, dim])
        # dim = functools.reduce(operator.iconcat, dim, [])

    dim_x = np.array([[dim[0][1], dim[0][0]], [dim[1][0], dim[1][1]]])
    dim_y = np.array([[dim[1][0], dim[0][0]], [dim[0][1], dim[1][1]]])

    x_tmp = swap(input_mat, [1, 2], dim, True)
    y_tmp = partial_transpose(x_tmp, sys=1, dim=dim_x)
    return swap(y_tmp, [1, 2], dim_y, True)


def reduction_map(dim: int, k: int = 1) -> np.ndarray:
    r"""
    Produce the reduction map.

    If `k = 1`, this returns the Choi matrix of the reduction map which is a
    positive map on `dim`-by-`dim` matrices. For a different value of `k`, this
    yields the Choi matrix of the map defined by:

    .. math::
        R(X) = k * \text{Tr}(X) * \mathbb{I} - X,

    where :math:`\mathbb{I}` is the identity matrix. This map is
    :math:`k`-positive.

    Examples
    ==========

    Using `toqito`, we can generate the $3$-dimensional (or standard) reduction
    map as follows.

    >>> from toqito.channels import reduction_map
    >>> reduction_map(3)
    [[ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.],
     [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
     [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
     [-1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
     [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
     [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
     [-1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.]])

    :param dim: A positive integer (the dimension of the reduction map).
    :param k:  If this positive integer is provided, the script will instead
               return the Choi matrix of the following linear map:
               Phi(X) := K * Tr(X)I - X.
    :return: The reduction map.
    """
    psi = max_entangled(dim, False, False)
    return k * identity(dim ** 2) - psi * psi.conj().T
