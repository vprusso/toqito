"""Kronecker tensor product of two or more matrices."""
from typing import List
import numpy as np


def tensor(input_1: np.ndarray, input_2: np.ndarray) -> np.ndarray:
    r"""
    Tensor product between two matrices.

    Tensor two matrices or vectors together using the standard kronecker
    operation provided from numpy.

    Given two matrices :math:`A` and :math:`B`, computes :math:`A \otimes B`.
    The same concept also applies to two vectors :math:`v` and :math:`w` which
    computes :math: `v \otimes w`.

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
