"""Kronecker tensor product of two or more matrices."""
from typing import List
import numpy as np


def tensor(input_1: np.ndarray, input_2: np.ndarray) -> np.ndarray:
    r"""
    Tensor product between two matrices [WIKTEN]_.

    Tensor two matrices or vectors together using the standard kronecker
    operation provided from numpy.

    Given two matrices :math:`A` and :math:`B`, computes :math:`A \otimes B`.
    The same concept also applies to two vectors :math:`v` and :math:`w` which
    computes :math: `v \otimes w`.

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

    >>> from toqito.core.ket import ket
    >>> from toqito.linear_algebra.operations.tensor import tensor
    >>> e_0 = ket(2, 0)
    >>> tensor(e_0, e_0)
    array([[1],
           [0],
           [0],
           [0]])

    References
    ==========
    .. [WIKTEN] Wikipedia: Tensor product
        https://en.wikipedia.org/wiki/Tensor_product

    :param input_1: The first matrix argument.
    :param input_2: The second matrix argument.
    :return: The tensor product between `input_1` and `input_2`.
    """
    return np.kron(input_1, input_2)


def tensor_n(input_val: np.ndarray, num_tensor: int) -> np.ndarray:
    r"""
    Tensor product one matrix `n` times with itself.

    For a matrix, :math:`A` and an integer :math:`n`, the result of this
    function computes :math:`A^{\otimes n}`.

    Similarly for a vector :math:`v` and an integer :math:`n`, the result of
    of this function computes :math:`v^{\otimes n}`.

    Tensor product one matrix :math:`n` times with itself.

    We may also tensor some element with itself some integer number of times.
    For instance we can compute

    .. math::
        |0 \rangle^{\otimes 3} = \left[1, 0, 0, 0, 0, 0, 0, 0 \right]^{\text{T}}

    in `toqito` as follows.

    >>> from toqito.core.ket import ket
    >>> from toqito.linear_algebra.operations.tensor import tensor_n
    >>> e_0 = ket(2, 0)
    >>> tensor_n(e_0, 3)
    array([[1],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0]])

    :param input_val: The matrix argument.
    :param num_tensor: The number of times to tensor.
    :return: The matrix `input_val` tensored with itself `num_tensor` times.
    """
    result = None
    if num_tensor == 1:
        return input_val
    if num_tensor == 2:
        return np.kron(input_val, input_val)
    if num_tensor >= 3:
        result = np.kron(input_val, input_val)
        for _ in range(2, num_tensor):
            result = np.kron(result, input_val)
    return result


def tensor_list(input_list: List[np.ndarray]) -> np.ndarray:
    r"""
    Perform the tensor product on a list of matrices.

    Given a list of :math:`n` matrices :math:`A_1, A_2, \ldots, A_n` the result
    of this function computes :math:`A_1 \otimes A_2 \otimes \ldots
    \otimes A_n`.

    Similarly, for a list of :math:`n` vectors :math:`v_1, v_2, \ldots, v_n`,
    the result of this function computes :math:`v_1 \otimes v_2 \otimes \ldots
    \otimes v_n`.

    Perform the tensor product on a list of vectors or matrices.

    If we wish to compute the tensor product against more than two matrices or
    vectors, we can feed them in as a `list`. For instance, if we wish to
    compute :math:`|0 \rangle \otimes |1 \rangle \otimes |0 \rangle`, we can do
    so as follows.

    >>> from toqito.core.ket import ket
    >>> from toqito.linear_algebra.operations.tensor import tensor_list
    >>> e_0, e_1 = ket(2, 0), ket(2, 1)
    >>> tensor_list([e_0, e_1, e_0])
    array([[0],
           [0],
           [1],
           [0],
           [0],
           [0],
           [0],
           [0]])

    :param input_list: A list of matrices.
    :return: The tensor product of all matrices in the list.
    """
    result = None
    if len(input_list) == 1:
        return input_list[0]
    if len(input_list) == 2:
        return np.kron(input_list[0], input_list[1])
    if len(input_list) >= 3:
        result = input_list[0]
        for i in range(1, len(input_list)):
            result = np.kron(result, input_list[i])
    return result
