"""Test tensor."""

import numpy as np
import pytest

from toqito.matrix_ops import tensor
from toqito.states import basis

e_0, e_1 = basis(2, 0), basis(2, 1)
matrix1 = np.array([[1, 2]])
matrix2 = np.array([[3], [4]])
matrix3 = np.array([[5, 6]])
matrix4 = np.array([[7, 8]])


@pytest.mark.parametrize(
    "test_input, len_input, expected",
    [
        # standard tensor product on vectors
        ((e_0, e_0), 2, np.kron(e_0, e_0)),
        # tensor product of 1 item to should return the item
        ([np.array([[1, 2], [3, 4]])], 1, np.array([[1, 2], [3, 4]])),
        # tensor product of multiple args as input
        ((np.identity(2), np.identity(2), np.identity(2), np.identity(2)), 4, np.identity(16)),
        # tensor product of array of 2 arrays
        (
            np.array([np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]),
            1,
            np.array([[5, 6, 10, 12], [7, 8, 14, 16], [15, 18, 20, 24], [21, 24, 28, 32]]),
        ),
        # tensor product of vector with n = 0
        ((e_0, 0), 2, np.array([[1.0]])),
        # tensor product of vector with n = 1
        ((e_0, 1), 2, e_0),
        # tensor product of vector with n = 2
        ((e_0, 2), 2, np.kron(e_0, e_0)),
        # tensor product of vector with n = 3
        ((e_0, 3), 2, np.kron(np.kron(e_0, e_0), e_0)),
        # tensor product of vector with n = 3
        ((e_0, 4), 2, np.kron(np.kron(np.kron(e_0, e_0), e_0), e_0)),
        # tensor product of empty list
        ([], 1, None),
        # tensor product of list with one item
        ([e_0], 1, e_0),
        # tensor product of list with two items
        ([e_0, e_1], 1, np.kron(e_0, e_1)),
        # tensor product of list with three items
        ([e_0, e_1, e_0], 1, np.kron(np.kron(e_0, e_1), e_0)),
        # tensor product of array of 3 arrays of identity matrices
        (np.array([np.identity(2), np.identity(2), np.identity(2)]), 1, np.identity(8)),
        # ((np.array([np.identity(2), np.identity(2), np.identity(2)])), 1, np.identity(8)),
        # tensor product of array of 4 arrays of identity matrices
        (np.array([np.identity(2), np.identity(2), np.identity(2), np.identity(2)]), 1, np.identity(16)),
        # tensor product with a numpy array containing three or more matrices
        (
            np.array([matrix1, matrix2, matrix3, matrix4], dtype=object),
            1,
            np.kron(np.kron(matrix1, np.kron(matrix2, matrix3)), matrix4),
        ),
        # tensor product of 1 matrix inside a list
        ([np.array([np.identity(4)])], 1, np.identity(4)),
    ],
)
def test_tensor_multiple_input(test_input, len_input, expected):
    """Test function works as expected."""
    if len_input == 1:
        calculated = tensor(test_input)
        assert calculated is expected or (calculated == expected).all()
    elif len_input == 2:
        calculated = tensor(test_input[0], test_input[1])
        assert calculated is expected or (calculated == expected).all()
    elif len_input == 4:
        calculated = tensor(test_input[0], test_input[1], test_input[2], test_input[3])
        assert (calculated == expected).all()


def test_tensor_empty_args():
    r"""Test tensor with no arguments."""
    with pytest.raises(ValueError, match="The `tensor` function must take either a matrix or vector."):
        tensor()
