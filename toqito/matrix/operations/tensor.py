"""Kronecker tensor product of two or more matrices."""
from typing import List
import numpy as np


def tensor(input_1: np.ndarray, input_2: np.ndarray) -> np.ndarray:
    """
    Tensor two matrices or vectors together using the standard kronecker
    operation provided from numpy.

    :param input_1:
    :param input_2:
    :return:
    """
    return np.kron(input_1, input_2)


def tensor_n(input_val: np.ndarray, num_tensor: int) -> np.ndarray:
    """
    :param input_val:
    :param num_tensor:
    :return:
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
    """
    :param input_list:
    :return:
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
