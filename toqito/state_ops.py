"""Operations on quantum states."""
from typing import List, Tuple, Union
from scipy.sparse import issparse, linalg

import numpy as np


__all__ = ["pure_to_mixed", "schmidt_decomposition", "tensor"]


def pure_to_mixed(phi: np.ndarray) -> np.ndarray:
    r"""
    Convert a state vector or density matrix to a density matrix.

    Examples
    ==========

    It is possible to convert a pure state vector to a mixed state vector using
    the `toqito` package. Consider the following Bell state

    .. math::
        u = \frac{1}{\sqrt{2}} \left( |00 \rangle + |11 \rangle \right).

    The corresponding mixed state from :math:`u` is calculated as

    .. math::
        \rho = u u^* = \frac{1}{2} \begin{pmatrix}
                                        1 & 0 & 0 & 1 \\
                                        0 & 0 & 0 & 0 \\
                                        0 & 0 & 0 & 0 \\
                                        1 & 0 & 0 & 1
                                   \end{pmatrix}

    Using `toqito`, we can obtain this matrix as follows.

    >>> from toqito.states import bell
    >>> from toqito.state_ops import pure_to_mixed
    >>> phi = bell(0)
    >>> pure_to_mixed(phi)
    [[0.5, 0. , 0. , 0.5],
     [0. , 0. , 0. , 0. ],
     [0. , 0. , 0. , 0. ],
     [0.5, 0. , 0. , 0.5]]

    We can also give matrix inputs to the function in `toqito`.

    >>> from toqito.states import bell
    >>> from toqito.state_ops import pure_to_mixed
    >>> phi = bell(0) * bell(0).conj().T
    >>> pure_to_mixed(phi)
    [[0.5, 0. , 0. , 0.5],
     [0. , 0. , 0. , 0. ],
     [0. , 0. , 0. , 0. ],
     [0.5, 0. , 0. , 0.5]])

    :param phi: A density matrix or a pure state vector.
    :return: density matrix representation of `phi`, regardless of
             whether `phi` is itself already a density matrix or if
             if is a pure state vector.
    """
    # Compute the size of `phi`. If it's already a mixed state, leave it alone.
    # If it's a vector (pure state), make it into a density matrix.
    row_dim, col_dim = phi.shape[0], phi.shape[1]

    # It's a pure state vector.
    if min(row_dim, col_dim) == 1:
        return phi * phi.conj().T
    # It's a density matrix.
    if row_dim == col_dim:
        return phi
    # It's neither.
    raise ValueError("InvalidDim: `phi` must be either a vector or square " "matrix.")


def schmidt_decomposition(
    vec: np.ndarray, dim: Union[int, List[int], np.ndarray] = None, k_param: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Compute the Schmidt decomposition of a bipartite vector [WikSD]_.

    Examples
    ==========

    Consider the :math:`3`-dimensional maximally entangled state

    .. math::
        u = \frac{1}{\sqrt{3}} \left( |000 \rangle +
        |111 \rangle + |222 \rangle \right)

    We can generate this state using the `toqito` module as follows.

    >>> from toqito.states import max_entangled
    >>> max_entangled(3)
    [[0.57735027],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.57735027],
     [0.        ],
     [0.        ],
     [0.        ],
     [0.57735027]]

    Computing the Schmidt decomposition of :math:`u`, we can obtain the
    corresponding singular values of :math:`u` as

    .. math::
        \frac{1}{\sqrt{3}} \left[1, 1, 1 \right]^{\text{T}}

    >>> from toqito.states import max_entangled
    >>> from toqito.state_ops import schmidt_decomposition
    >>> singular_vals, u_mat, vt_mat = schmidt_decomposition(max_entangled(3))
    >>> singular_vals
    [[0.57735027]
     [0.57735027]
     [0.57735027]]
    >>> u_mat
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    >>> vt_mat
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]

    References
    ==========
    .. [WikSD] Wikipedia: Schmidt decomposition
        https://en.wikipedia.org/wiki/Schmidt_decomposition

    :param vec:
    :param dim:
    :param k_param:
    :return: The Schmidt decomposition of the `vec` input.
    """
    eps = np.finfo(float).eps

    if dim is None:
        dim = np.round(np.sqrt(len(vec)))
    if isinstance(dim, list):
        dim = np.array(dim)

    # Allow the user to enter a single number for `dim`.
    if isinstance(dim, float):
        dim = np.array([dim, len(vec) / dim])
        if np.abs(dim[1] - np.round(dim[1])) >= 2 * len(vec) * eps:
            raise ValueError(
                "InvalidDim: The value of `dim` must evenly divide"
                " `len(vec)`; please provide a `dim` array "
                "containing the dimensions of the subsystems."
            )
        dim[1] = np.round(dim[1])

    # Try to guess whether SVD or SVDS will be faster, and then perform the
    # appropriate singular value decomposition.
    adj = 20 + 1000 * (not issparse(vec))

    # Just a few Schmidt coefficients.
    if 0 < k_param <= np.ceil(np.min(dim) / adj):
        u_mat, singular_vals, vt_mat = linalg.svds(
            linalg.LinearOperator(np.reshape(vec, dim[::-1].astype(int)), k_param)
        )
    # Otherwise, use lots of Schmidt coefficients.
    else:
        u_mat, singular_vals, vt_mat = np.linalg.svd(
            np.reshape(vec, dim[::-1].astype(int))
        )

    if k_param > 0:
        u_mat = u_mat[:, :k_param]
        singular_vals = singular_vals[:k_param]
        vt_mat = vt_mat[:, :k_param]

    # singular_vals = np.diag(singular_vals)
    singular_vals = singular_vals.reshape(-1, 1)
    if k_param == 0:
        # Schmidt rank.
        r_param = np.sum(singular_vals > np.max(dim) * np.spacing(singular_vals[0]))
        # Schmidt coefficients.
        singular_vals = singular_vals[:r_param]
        u_mat = u_mat[:, :r_param]
        vt_mat = vt_mat[:, :r_param]

    u_mat = u_mat.conj().T
    return singular_vals, u_mat, vt_mat


def tensor(*args) -> np.ndarray:
    r"""
    Compute the Kronecker tensor product [WikTensor]_.

    Tensor two matrices or vectors together using the standard Kronecker
    operation provided from numpy.

    Given two matrices :math:`A` and :math:`B`, computes :math:`A \otimes B`.
    The same concept also applies to two vectors :math:`v` and :math:`w` which
    computes :math:`v \otimes w`.

    One may also compute the tensor product one matrix `n` times with itself.

    For a matrix, :math:`A` and an integer :math:`n`, the result of this
    function computes :math:`A^{\otimes n}`.

    Similarly for a vector :math:`v` and an integer :math:`n`, the result of
    of this function computes :math:`v^{\otimes n}`.

    One may also perform the tensor product on a list of matrices.

    Given a list of :math:`n` matrices :math:`A_1, A_2, \ldots, A_n` the result
    of this function computes

    .. math::
        A_1 \otimes A_2 \otimes \ldots \otimes A_n.

    Similarly, for a list of :math:`n` vectors :math:`v_1, v_2, \ldots, v_n`,
    the result of this function computes

    .. math::
        v_1 \otimes v_2 \otimes \ldots \otimes v_n.

    Examples
    ==========

    Tensor product two matrices or vectors

    Consider the following ket vector

    .. math::
        |0 \rangle = \left[1, 0 \right]^{\text{T}}

    Computing the following tensor product

    .. math:
        |0 \rangle \otimes |0 \rangle = \left[1, 0, 0, 0 \right]^{\text{T}}

    This can be accomplished in `toqito` as follows.

    >>> from toqito.states import basis
    >>> from toqito.state_ops import tensor
    >>> e_0 = basis(2, 0)
    >>> tensor(e_0, e_0)
    [[1],
     [0],
     [0],
     [0]]

    Tensor product one matrix :math:`n` times with itself.

    We may also tensor some element with itself some integer number of times.
    For instance we can compute

    .. math::
        |0 \rangle^{\otimes 3} = \left[1, 0, 0, 0, 0, 0, 0, 0 \right]^{\text{T}}

    in `toqito` as follows.

    >>> from toqito.states import basis
    >>> from toqito.state_ops.tensor import tensor
    >>> e_0 = basis(2, 0)
    >>> tensor(e_0, 3)
    [[1],
     [0],
     [0],
     [0],
     [0],
     [0],
     [0],
     [0]]

    Perform the tensor product on a list of vectors or matrices.

    If we wish to compute the tensor product against more than two matrices or
    vectors, we can feed them in as a `list`. For instance, if we wish to
    compute :math:`|0 \rangle \otimes |1 \rangle \otimes |0 \rangle`, we can do
    so as follows.

    >>> from toqito.states import basis
    >>> from toqito.state_ops.tensor import tensor
    >>> e_0, e_1 = basis(2, 0), basis(2, 1)
    >>> tensor([e_0, e_1, e_0])
    [[0],
     [0],
     [1],
     [0],
     [0],
     [0],
     [0],
     [0]]

    References
    ==========
    .. [WikTensor] Wikipedia: Tensor product
        https://en.wikipedia.org/wiki/Tensor_product

    :param args: Input to the tensor function is expected to be either:
        - List[np.ndarray]: List of numpy matrices,
        - np.ndarray, ... np.ndarray: An arbitrary number of numpy arrays,
        - np.ndarray, int: A numpy array and an integer.
    :return: The computed tensor product.
    """
    result = None

    # Input is provided as a list of matrices.
    if len(args) == 1 and isinstance(args[0], list):
        if len(args[0]) == 1:
            return args[0][0]
        if len(args[0]) == 2:
            return np.kron(args[0][0], args[0][1])
        if len(args[0]) >= 3:
            result = args[0][0]
            for i in range(1, len(args[0])):
                result = np.kron(result, args[0][i])
        return result

    # Tensor product one matrix `n` times with itself.
    if len(args) == 2 and isinstance(args[1], int):
        num_tensor = args[1]
        if num_tensor == 1:
            return args[0]
        if num_tensor == 2:
            return np.kron(args[0], args[0])
        if num_tensor >= 3:
            result = np.kron(args[0], args[0])
            for _ in range(2, num_tensor):
                result = np.kron(result, args[0])
        return result

    # Tensor product between two or more matrices.
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        return np.kron(args[0], args[1])
    if len(args) >= 3:
        result = args[0]
        for i in range(1, len(args)):
            result = np.kron(result, args[i])
        return result
    raise ValueError("The `tensor` function must take either a matrix or " "vector.")
