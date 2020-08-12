"""Tensor product operation."""
import numpy as np


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

    in `toqito` as follows.

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
    .. [WikTensor] Wikipedia: Tensor product
        https://en.wikipedia.org/wiki/Tensor_product

    :param args: Input to the tensor function is expected to be either:
        - List[np.ndarray]: List of numpy matrices,
        - np.ndarray, ... , np.ndarray: An arbitrary number of numpy arrays,
        - np.ndarray, int: A numpy array and an integer.
    :return: The computed tensor product.
    """
    result = None

    # Input is provided as a list of numpy matrices.
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

    if len(args) == 1 and isinstance(args[0], np.ndarray):
        # If the numpy array is just a single matrix, so the dimensions are
        # provided as an (x, y)-tuple.
        if len(args[0].shape) == 2:
            return args[0]
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
    if len(args) == 2:
        return np.kron(args[0], args[1])
    if len(args) >= 3:
        result = args[0]
        for i in range(1, len(args)):
            result = np.kron(result, args[i])
        return result
    raise ValueError("The `tensor` function must take either a matrix or vector.")
