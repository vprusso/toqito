import numpy as np
from typing import List


def tensor(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.kron(A, B)


def tensor_n(A: np.ndarray, n: int) -> np.ndarray:
    result = None
    if n == 1:
        return A
    if n == 2:
        return np.kron(A, A)
    if n >= 3:
        result = np.kron(A, A)
        for _ in range(2, n):
            result = np.kron(result, A)
    return result


def tensor_list(input_list: List[np.ndarray]) -> np.ndarray:
    result = None
    if len(input_list) == 1:
        return input_list[0]
    elif len(input_list) == 2:
        return np.kron(input_list[0], input_list[1])
    elif len(input_list) >= 3:
        result = np.kron(input_list[0], input_list[1])
        for i in range(2, len(input_list)):
            result = np.kron(result, input_list[i])
    return result
