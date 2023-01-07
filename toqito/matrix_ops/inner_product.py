"""Inner product operation"""
import numpy as np

def inner_product(v1: np.ndarray, v2: np.ndarray) -> float:
    r"""
    Compute the inner product of two vectors [WikInner]_.

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
        A_1 \otimes A_2 \otimes \cdots \otimes A_n.

    Similarly, for a list of :math:`n` vectors :math:`v_1, v_2, \ldots, v_n`,
    the result of this function computes

    .. math::
        v_1 \otimes v_2 \otimes \cdots \otimes v_n.

    Examples
    ==========

    Tensor product two matrices or vectors

    Consider the following ket vector

    .. math::
        e_0 = \left[1, 0 \right]^{\text{T}}.

    Computing the following tensor product

    .. math:
        e_0 \otimes e_0 = \[1, 0, 0, 0 \]^{\text{T}}.

    This can be accomplished in :code:`toqito` as follows.

    >>> from toqito.states import basis
    >>> from toqito.matrix_ops import tensor
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
        e_0^{\otimes 3} = \left[1, 0, 0, 0, 0, 0, 0, 0 \right]^{\text{T}}

    in :code:`toqito` as follows.

    >>> from toqito.states import basis
    >>> from toqito.matrix_ops import tensor
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
    compute :math:`e_0 \otimes e_1 \otimes e_0`, we can do
    so as follows.

    >>> from toqito.states import basis
    >>> from toqito.matrix_ops import tensor
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
    .. [WikInner] Wikipedia: Inner product space
        https://en.wikipedia.org/wiki/Inner_product_space

    :raises ValueError: Vector dimenstions are mismatched.
    :param args: v1 and v2, both vectors of dimenstions :math:`(n,1)` where n>1.
    :return: The computed inner product.
    """

    # Check for dimensional validity
    if not (v1.shape[0] == v2.shape[0] and v1.shape[0] > 1 and v1.shape[1] == v2.shape[1] == 1):
        raise ValueError("Dimension mismatch")
    
    res = 0
    for i in range(v1.shape[0]):
        res += v1[i,0] * v2[i,0]
    
    return res